from django.contrib import admin
from stamps.models import Series, Country, Stamp, Format, Emission

admin.site.register(Series)
admin.site.register(Country)
admin.site.register(Stamp)
admin.site.register(Format)
admin.site.register(Emission)
