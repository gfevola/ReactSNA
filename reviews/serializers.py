#completeversion/data/serializers
from .models import ReviewModel, Review, ReviewBigrams, WordScores
from rest_framework import serializers


class WordsSerializer(serializers.ModelSerializer):
    class Meta:
        model = WordScores
        fields = "__all__"

    
class ReviewSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = Review
        fields = ("ModelKey","ReviewID","ReviewItem","Summary","Description","Rating","Topic","TopicScore",
                    'Rating_Estimate',"SentimentTrend","WordTrend")


class ReviewBigramSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReviewBigrams
        fields = "__all__"


#Model-level serializers
#only Model
class ReviewModelOnlySerializer(serializers.ModelSerializer):
    
    class Meta:
        model = ReviewModel
        fields = "__all__"

#with cascaded models 
class ReviewModelSerializer(serializers.ModelSerializer):
    
    reviews = ReviewSerializer(read_only=True,source="review_encrypt",many=True)
    words = WordsSerializer(read_only=True,source="word_encrypt",many=True)
    
    class Meta:
        model = ReviewModel
        fields = ( 
            'ModelKey','ModelType','reviews', 'words'
        )
        
#model w bigrams        
class ReviewModelBG_Serializer(serializers.ModelSerializer):
    
    bigrams = ReviewBigramSerializer(read_only=True,source="bigram_encrypt",many=True)
    
    class Meta:
        model = ReviewModel
        fields = ( 
            'ModelKey','ModelType','bigrams'
        )
        