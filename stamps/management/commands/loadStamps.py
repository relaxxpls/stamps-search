from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from django.core.management.base import BaseCommand
from stamps.models import Country, Series, Stamp, Format, Emission
from tqdm import tqdm
from django.utils.dateparse import parse_date
from django.db import transaction


def extract_sg_code(text: str):
    parts = text.split(",")
    parts = [part.strip() for part in parts]
    for part in parts:
        if part.startswith("Sg:"):
            return part.split("Sg:")[1].strip()
    return None


def parse_partial_date(date_string: str):
    parsed_date = parse_date(date_string.strip())
    if parsed_date:
        return parsed_date

    # ? If full date parsing fails, try parsing as a partial date
    if len(date_string) == 7:  # YYYY-MM format
        try:
            return datetime.strptime(date_string, "%Y-%m").date().replace(day=1)
        except ValueError:
            date_string = date_string.split("-")[0]

    if len(date_string) == 4:  # YYYY format
        return datetime.strptime(date_string, "%Y").date().replace(month=1, day=1)

    return None


def process_chunk(chunk):
    countries = {}
    series = {}
    emissions = {}
    formats = {}
    stamps = []

    for row in chunk.itertuples():
        if row.country not in countries:
            countries[row.country] = Country(name=row.country)
        if row.format not in formats:
            formats[row.format] = Format(name=row.format)

        if pd.notna(row.series) and row.series not in series:
            series[row.series] = Series(name=row.series)
        if pd.notna(row.emission) and row.emission not in emissions:
            emissions[row.emission] = Emission(name=row.emission)

        sg_code = extract_sg_code(row.catalog_codes)
        if sg_code is None:
            print(f"Skipping row {row.Index}. {row.colnect_url} with missing SG code")
            continue

        stamp = Stamp(
            id=row.colnect_id,
            colnect_url=row.colnect_url,
            catalog_code_sg=sg_code,
            image_url=row.image_url,
            title=row.title,
            country=countries.get(row.country, None),
            series=series.get(row.series, None),
            description=row.description if pd.notna(row.description) else "",
            issued_on=parse_partial_date(row.issued_on),
            face_value=row.face_value if pd.notna(row.face_value) else "",
            emission=emissions.get(row.emission, None),
            format=formats.get(row.format, None),
        )
        stamps.append(stamp)

    return countries, series, emissions, formats, stamps


class Command(BaseCommand):
    help = "Loads stamps data from a CSV file"
    chunk_size = 10000  # Adjust based on your system's capabilities
    num_workers = 4  # Adjust based on your CPU cores

    def add_arguments(self, parser):
        parser.add_argument("csv", nargs=1)

    def handle(self, *_, **options):
        print("Processing data chunks")
        csv_file = Path(options["csv"][0]).resolve()

        if not csv_file.exists():
            raise ValueError("CSV file not found")

        df = pd.read_csv(csv_file)
        # ? First 32 csv entries are duplicates
        df = df.drop_duplicates(subset=["colnect_url"], keep="last")

        chunks = np.array_split(df, len(df) // self.chunk_size + 1)
        total_chunks = len(chunks)

        with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
            for chunk in chunks:
                countries, series, emissions, formats, stamps = process_chunk(chunk)
                with transaction.atomic():
                    Country.objects.bulk_create(
                        countries.values(), ignore_conflicts=True
                    )
                    Series.objects.bulk_create(series.values(), ignore_conflicts=True)
                    Emission.objects.bulk_create(
                        emissions.values(), ignore_conflicts=True
                    )
                    Format.objects.bulk_create(formats.values(), ignore_conflicts=True)

                    country_map = {
                        c.name: c
                        for c in Country.objects.filter(name__in=countries.keys())
                    }
                    series_map = {
                        s.name: s for s in Series.objects.filter(name__in=series.keys())
                    }
                    emissions_map = {
                        e.name: e
                        for e in Emission.objects.filter(name__in=emissions.keys())
                    }
                    format_map = {
                        f.name: f
                        for f in Format.objects.filter(name__in=formats.keys())
                    }

                    for stamp in stamps:
                        stamp.country = country_map[stamp.country.name]
                        stamp.series = (
                            series_map[stamp.series.name] if stamp.series else None
                        )
                        stamp.emission = (
                            emissions_map[stamp.emission.name]
                            if stamp.emission
                            else None
                        )
                        stamp.format = format_map[stamp.format.name]

                    Stamp.objects.bulk_create(stamps)
                pbar.update(1)

        print("Data import completed successfully")
