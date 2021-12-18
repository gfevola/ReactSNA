#completeversion/data/views
from django.shortcuts import render
from .serializers import DataSerializer
from .models import EmpData
from rest_framework import viewsets
from rest_framework.generics import GenericAPIView, ListAPIView, RetrieveAPIView
from rest_framework import permissions
from rest_framework.views import APIView
from rest_framework.response import Response


# Create your views here.
class DataView(ListAPIView):
    permission_classes = (permissions.AllowAny, )
    serializer_class = DataSerializer
    queryset = EmpData.objects.filter(location="N/A")


class DataViewPost(APIView):
    permission_classes = (permissions.AllowAny, )

    def post(self, request, format=None):
        data = self.request.data
        queryset = EmpData.objects.filter(location=data['location'])    
        serializer_class = DataSerializer(queryset, many=True)
        return Response(serializer_class.data)    
        
#this is not working
class DataUpload(APIView):
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
 