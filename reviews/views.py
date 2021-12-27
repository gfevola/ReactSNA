#completeversion/reviews/views
from django.shortcuts import render
from django.conf import settings
from django.contrib import messages
from .models import ReviewModel, Review, ReviewBigrams, WordScores
from rest_framework.views import APIView
from rest_framework import permissions
from rest_framework.response import Response

from data.models import Document, EmpData
from .serializers import ReviewModelSerializer, ReviewModelOnlySerializer, ReviewModelBG_Serializer

import pandas as pd

import sys
sys.path.insert(0, './reviews/pythonscripts')
import Bigrams_from_Reviews as bg
import Review_Topics as rt
import Prepare_Reviews_NLTK as pr
import Review_Description_Sentiment_Analysis as ds


class ReviewViewPost(APIView):
#view full data for specific model name
    permission_classes = (permissions.AllowAny, )
    
    def post(self, request, format=None):
        data = self.request.data
        print('reviewmodel')
        queryset = ReviewModel.objects.filter(ModelKey=data['modelname'])    
        serializer_class = ReviewModelSerializer(queryset, many=True)
        return Response(serializer_class.data)  
 

class ReviewModelsPost(APIView):
#view a list of review model names
    permission_classes = (permissions.AllowAny, )
    
    def post(self, request, format=None):
        data = self.request.data
        queryset = ReviewModel.objects.all() 
        serializer_class = ReviewModelOnlySerializer(queryset, many=True)
        return Response(serializer_class.data)  

 
class ReviewModelBigram_View(APIView):
#view model -> bigrams 
    permission_classes = (permissions.AllowAny, )
    
    def post(self, request, format=None):
        data = self.request.data
        queryset = ReviewModel.objects.filter(ModelKey=data['modelname'])    
        serializer_class = ReviewModelBG_Serializer(queryset, many=True)
        return Response(serializer_class.data)  
 
class DeleteModelsPost(APIView):
#view a list of review model names
    permission_classes = (permissions.AllowAny, )
    
    def post(self, request, format=None):
        data = self.request.data
        queryset1 = ReviewModel.objects.filter(ModelKey=data['modelname'])
        queryset1.delete()
        return Response({})  
 

class ReviewDataImport(APIView):

    permission_classes = (permissions.AllowAny, )

    def post(self, request, format=None):
        data = self.request.data
        UploadReviews(request, data['reviewFile'], data['modelname'] )
        return Response({'note':'imported successfully'})

###this is working
class DataUpload1(APIView):
    permission_classes = (permissions.AllowAny, )

    def post(self, request, format=None):
        data = request.FILES['datafile']
        UploadData(data)
        return Response({'note':'imported successfully'})


def UploadData(file):
    newfile = Document(docfile = file)
    newfile.save()
    path = settings.MEDIA_ROOT.replace("\\","/") + "/" + str(newfile.docfile)
    DataFile = pd.read_excel(path)   
    
    for index, row in DataFile.iterrows():
        foo = EmpData(
             name = row['Name'],
             empid = row['EmpID'],
             location = row['Location']
        )
        foo.save()    
 
#----------------------------------------------------
# def ReviewTemplate(request):
# #template view - upload reviews
    # context = {}
    # if request.method == 'POST':
        # filename = request.FILES['reviewFile']
        # modelName = request.POST['modelname']
        
        # UploadReviews(request, filename,modelName)
    
    # return render(request,
                # 'reviews/upload.html',context
           # )

#--upload function
def UploadReviews(request,file,modelName):
    
    newfile = Document(docfile = file)
    newfile.save()
    
    path = settings.MEDIA_ROOT.replace("\\","/") + "/" + str(newfile.docfile)
    DataFile = pd.read_excel(path)
    #delete history
    try:
        history = ReviewModel.objects.get(ModelKey = modelName)
        history.delete()
        print("deleted history")
    except:
        pass
    
    foo = ReviewModel(
            ModelKey = modelName,
            ModelType = "Reviews"
    )
    foo.save()

    model = ReviewModel.objects.get(ModelKey = modelName)

    #converted description to bigrams (for graph)
    Reviews = pr.PrepareReviews(DataFile,"Text")
    Bigrams = bg.BigramCreate(Reviews,"ProductId","Score")
    [Reviews, Bigrams, WordTopics] = rt.TopicCreate(Reviews, Bigrams)  
    [Reviews, WordSentiment] = ds.DescriptionSentiment(Reviews, "Text","Score")
    
    WordDF = pd.merge(WordSentiment,WordTopics,left_on="Term",right_on="Term",how="inner")
    
    for index, row in Bigrams.iterrows():
        foo = ReviewBigrams(
                ModelKey = model,
                Bigram_Term1 = row['Term1'],
                Bigram_Term2 = row['Term2'],
                Score = row['Score'],
                Distance = 0,
                Count_Term1 = row['Count_x'],
                Count_Term2 = row['Count_y'],
                Term_Num1 = row['Num_x'],
                Term_Num2 = row['Num_y'],
        )
        foo.save()
    print('Completed Bigrams Upload')

    #reviews file
    for index, row in Reviews.iterrows():
        foo = Review(
                ModelKey = model,
                ReviewID = row['UniqueID'],
                ReviewItem = row['ProductId'],
                Summary = row['Summary'],
                Description = row['Text'],
                Rating = row['Score'],
                Topic = row['Topic'],
                TopicScore = row['TopicScore'],
                Rating_Estimate = row['EstimatedSentiment'],
                SentimentTrend = row['SentimentString'],
                WordTrend = row['Wordlist']
        )
        foo.save()
        
    print('Completed Reviews Upload')
    messages.success(request,f'Uploaded Reviews Successfully')

    #word scores
    for index, row in WordDF.iterrows():
        foo = WordScores(
            ModelKey = model,
            Word = row['Term'],
            WScore = row['SentimentScore'],
            POS_Tag = row['POS_Tag'],
            WTopicNum = row['Topic'],
            WTopicScore = row['TopicScore']
        )
        foo.save()
    print('Completed Words Upload')
    
    return({'fieldNames': DataFile.columns})
