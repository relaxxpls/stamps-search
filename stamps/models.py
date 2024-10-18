from django.db import models


class NameModel(models.Model):
    name = models.CharField(unique=True, max_length=255)

    class Meta:
        abstract = True

    def __str__(self):
        return self.name


class Country(NameModel):
    pass


class Series(NameModel):
    pass


class Stamp(models.Model):
    """
    Model to store information about a postage stamp.
    """

    colnect_url = models.URLField(unique=True)
    catalog_code_sg = models.CharField(max_length=15, unique=True)
    image_url = models.URLField(blank=True)
    title = models.CharField(max_length=255)
    country = models.ForeignKey(Country, on_delete=models.CASCADE)
    series = models.ForeignKey(Series, on_delete=models.CASCADE)
    description = models.TextField(blank=True)
    issued_on = models.DateField(blank=True, null=True)
    face_value = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return self.name
