# Generated by Django 3.2.8 on 2021-10-18 14:50

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ReviewModel',
            fields=[
                ('ModelKey', models.CharField(default='N/A', max_length=100, primary_key=True, serialize=False)),
                ('ModelType', models.CharField(default='N/A', max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Review',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ReviewItem', models.CharField(default='N/A', max_length=200)),
                ('Summary', models.CharField(default='N/A', max_length=200)),
                ('Description', models.CharField(default='N/A', max_length=8000)),
                ('Rating', models.DecimalField(decimal_places=2, default=0, max_digits=4)),
                ('ModelKey', models.ForeignKey(max_length=100, on_delete=django.db.models.deletion.CASCADE, related_name='review_encrypt', to='reviews.reviewmodel')),
            ],
        ),
    ]
