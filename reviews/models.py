from django.db import models

class ReviewModel(models.Model):
    ModelKey = models.CharField(max_length=100,primary_key=True,default="N/A")
    ModelType = models.CharField(max_length=100,default="N/A")


class Review(models.Model):
    ModelKey = models.ForeignKey(ReviewModel, max_length=100,on_delete=models.CASCADE,related_name="review_encrypt")
    ReviewItem = models.CharField(max_length=200,default="N/A")
    Summary = models.CharField(max_length=200,default="N/A")
    Description = models.CharField(max_length=8000,default="N/A")
    Rating = models.DecimalField(max_digits=4,decimal_places=2,default=0)
    Topic = models.CharField(max_length=200,default="N/A")
    TopicScore = models.DecimalField(max_digits=4,decimal_places=4,default=0)


class ReviewBigrams(models.Model):
    ModelKey = models.ForeignKey(ReviewModel, max_length=100,on_delete=models.CASCADE,related_name="bigram_encrypt")
    Bigram_Term1 = models.CharField(max_length=200,default="N/A")
    Bigram_Term2 = models.CharField(max_length=200,default="N/A")
    Score = models.DecimalField(max_digits=8,decimal_places=4,default=0)
    Distance = models.DecimalField(max_digits=8,decimal_places=4,default=0)
    Count_Term1 = models.DecimalField(max_digits=10,decimal_places=2,default=0)
    Count_Term2 = models.DecimalField(max_digits=10,decimal_places=2,default=0)
    Term_Num1 = models.DecimalField(max_digits=10,decimal_places=2,default=0)
    Term_Num2 = models.DecimalField(max_digits=10,decimal_places=2,default=0)