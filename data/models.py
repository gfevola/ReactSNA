#completeversion/data/models
from django.db import models


class Document(models.Model):
    docfile = models.FileField(upload_to='documents')

class EmpData(models.Model):
    name = models.CharField(max_length=150)
    empid = models.CharField(max_length=150)
    location = models.CharField(max_length=150, default="N/A")

#class Review(models.Model):
        