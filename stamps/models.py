from django.db import models


class Country(models.Model):
    name = models.CharField(max_length=255, unique=True)


class Series(models.Model):
    name = models.CharField(max_length=255, unique=True)


class Stamp(models.Model):
    id = models.AutoField(primary_key=True)
    colnect_url = models.URLField(unique=True)
    catalog_code_sg = models.CharField(max_length=15, unique=True)
    image_url = models.URLField(blank=True)
    title = models.CharField(max_length=255)
    country = models.ForeignKey(Country, on_delete=models.CASCADE)
    series = models.ForeignKey(Series, on_delete=models.SET_NULL, null=True, blank=True)
    description = models.TextField(blank=True)
    issued_on = models.DateField(blank=True, null=True)
    face_value = models.CharField(max_length=100, blank=True)

    def __str__(self):
        return self.name
