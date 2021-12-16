#completeversion/data/serializers
from .models import EmpData
from rest_framework import serializers


class DataSerializer(serializers.ModelSerializer):
    class Meta:
        model = EmpData
        fields = ('empid', 'name', 'location')