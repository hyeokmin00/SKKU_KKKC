from django.db import models


# Create your models here.
class Organizations(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=20)
    image = models.CharField(max_length=255)
    call = models.CharField(max_length=50)
    work = models.CharField(max_length=100)
    url = models.CharField(max_length=255)
    category = models.CharField(max_length=20)
