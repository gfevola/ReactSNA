# Generated by Django 3.2.8 on 2021-12-24 14:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('reviews', '0012_auto_20211223_1941'),
    ]

    operations = [
        migrations.AlterField(
            model_name='review',
            name='Rating_Estimate',
            field=models.DecimalField(decimal_places=4, default=0, max_digits=5),
        ),
    ]
