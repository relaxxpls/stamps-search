from pathlib import Path
import pandas as pd
import re
from django.core.management.base import BaseCommand
from stamps.models import Country, Series, Stamp
from tqdm import tqdm
from django.utils.dateparse import parse_date
from django.db import transaction
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np


def extract_sg_code(text: str):
    match = re.search(r"Sg:(\w+\s+\d+)", text)
    return match.group(1) if match else None


def process_chunk(chunk):
    countries = {}
    series = {}
    stamps = []

    for row in chunk.itertuples():
        if row.country not in countries:
            countries[row.country] = Country(name=row.country)
        if row.series not in series:
            series[row.series] = Series(name=row.series)

        stamp = Stamp(
            id=row.Index,
            colnect_url=row.colnect_url,
            catalog_code_sg=extract_sg_code(row.catalog_codes),
            image_url=row.image_url,
            title=row.title,
            country=countries[row.country],
            series=series[row.series],
            description=row.description,
            issued_on=parse_date(row.issued_on),
            face_value=row.face_value,
        )
        stamps.append(stamp)

    return countries, series, stamps


class Command(BaseCommand):
    help = "Migrates, loades data, and creates superuser"
    chunk_size = 10000  # Adjust based on your system's capabilities
    num_workers = 4  # Adjust based on your CPU cores

    def add_arguments(self, parser):
        parser.add_argument("csv", nargs=1)

    def handle(self, *_, **options):
        csv_file = Path(options["csv"][0]).resolve()

        if not csv_file.exists():
            raise ValueError("Csv file not found")

        df = pd.read_csv(csv_file)

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []

            for chunk in tqdm(np.array_split(df, len(df) // (self.chunk_size + 1))):
                futures.append(executor.submit(process_chunk, chunk))

            for future in tqdm(as_completed(futures), total=len(futures)):
                countries, series, stamps = future.result()
                with transaction.atomic():
                    Country.objects.bulk_create(
                        countries.values(), ignore_conflicts=True
                    )
                    Series.objects.bulk_create(series.values(), ignore_conflicts=True)

                    country_map = {
                        c.name: c
                        for c in Country.objects.filter(name__in=countries.keys())
                    }
                    series_map = {
                        s.name: s for s in Series.objects.filter(name__in=series.keys())
                    }
                    for stamp in stamps:
                        stamp.country = country_map[stamp.country.name]
                        stamp.series = series_map[stamp.series.name]

                Stamp.objects.bulk_create(stamps)

        print("Data loaded successfully")
