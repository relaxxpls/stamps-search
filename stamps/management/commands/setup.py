from django.contrib.auth import get_user_model
from django.core import management
from django.core.management.base import BaseCommand

User = get_user_model()


class Command(BaseCommand):
    help = "Migrates, loades data, downloads resources, and creates superuser"

    def add_arguments(self, parser):
        parser.add_argument("csv", nargs=1)

    def handle(self, *args, **options):
        management.call_command("migrate")
        management.call_command("collectstatic")
        management.call_command("createsuperuser")

        management.call_command("loadStamps", options["csv"][0])
