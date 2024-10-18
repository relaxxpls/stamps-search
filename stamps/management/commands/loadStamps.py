from pathlib import Path
import pandas as pd
import re
from django.core.management.base import BaseCommand
from stamps.models import Country, Series, Stamp
from tqdm import tqdm
from django.utils.dateparse import parse_date


class Command(BaseCommand):
    help = "Migrates, loades data, and creates superuser"

    def add_arguments(self, parser):
        parser.add_argument("csv", nargs=1)

    def handle(self, *_, **options):
        csv_file = Path(options["csv"][0]).resolve()

        if not csv_file.exists():
            raise ValueError("Csv file not found")

        df = pd.read_csv(csv_file)

        for _, row in tqdm(df.iterrows()):
            country, _ = Country.objects.get_or_create(name=row["country"])
            series, _ = Series.objects.get_or_create(name=row["series"])

            stamp = Stamp()
            stamp.colnect_url = row["colnect_url"]
            stamp.catalog_code_sg = self.extract_sg_code(row["catalog_codes"])
            stamp.image_url = row["image_url"]
            stamp.title = row["title"]
            stamp.country = country
            stamp.series = series
            stamp.description = row["description"]
            stamp.issued_on = parse_date(row["issued_on"])
            stamp.face_value = row["face_value"]
            stamp.save()

    def extract_sg_code(text: str):
        match = re.search(r"Sg:(\w+\s+\d+)", text)

        if match:
            return match.group(1)
        else:
            return None
