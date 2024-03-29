# Generated by Django 3.2.8 on 2021-10-19 22:33

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('reviews', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ReviewBigrams',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Bigram_Term2', models.CharField(default='N/A', max_length=200)),
                ('Bigram_Term1', models.CharField(default='N/A', max_length=200)),
                ('Score', models.DecimalField(decimal_places=4, default=0, max_digits=4)),
                ('Distance', models.DecimalField(decimal_places=4, default=0, max_digits=4)),
                ('Count_Term1', models.DecimalField(decimal_places=2, default=0, max_digits=10)),
                ('Count_Term2', models.DecimalField(decimal_places=2, default=0, max_digits=10)),
                ('Term_Num1', models.DecimalField(decimal_places=2, default=0, max_digits=10)),
                ('Term_Num2', models.DecimalField(decimal_places=2, default=0, max_digits=10)),
                ('ModelKey', models.ForeignKey(max_length=100, on_delete=django.db.models.deletion.CASCADE, related_name='bigram_encrypt', to='reviews.reviewmodel')),
            ],
        ),
    ]
