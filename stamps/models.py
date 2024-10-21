from django.db import models


class Country(models.Model):
    name = models.CharField(max_length=127, unique=True)


class Series(models.Model):
    name = models.CharField(max_length=127, unique=True)


class Format(models.Model):
    name = models.CharField(max_length=31, unique=True)


class Emission(models.Model):
    name = models.CharField(max_length=31, unique=True)


class Stamp(models.Model):
    id = models.IntegerField(primary_key=True)
    colnect_url = models.URLField(max_length=255, unique=True)
    catalog_code_sg = models.CharField(max_length=31, unique=True)
    image_url = models.URLField(max_length=127)
    title = models.CharField()
    country = models.ForeignKey(Country, on_delete=models.CASCADE)
    series = models.ForeignKey(Series, on_delete=models.SET_NULL, null=True, blank=True)
    description = models.TextField(blank=True)
    issued_on = models.DateField()
    face_value = models.CharField(max_length=127)
    format = models.ForeignKey(Format, on_delete=models.CASCADE)
    emission = models.ForeignKey(
        Emission, on_delete=models.SET_NULL, null=True, blank=True
    )

    def __str__(self):
        return self.name
