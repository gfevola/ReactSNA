# Generated by Django 3.2.8 on 2021-12-26 21:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('reviews', '0013_alter_review_rating_estimate'),
    ]

    operations = [
        migrations.AddField(
            model_name='review',
            name='WordTrend',
            field=models.CharField(default='N/A', max_length=1000),
        ),
        migrations.AlterField(
            model_name='review',
            name='SentimentTrend',
            field=models.CharField(default='N/A', max_length=1000),
        ),
    ]