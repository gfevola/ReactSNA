#completeversion/data/admin
from django.contrib import admin
from .models import EmpData

class DataAdmin(admin.ModelAdmin):
    list_display = ('id', 'empid','name','location')

admin.site.register(EmpData, DataAdmin)
