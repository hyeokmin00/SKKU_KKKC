from django.db import models

# Create your models here.

# 유관기관 정보
class Organizations(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=20)
    image = models.CharField(max_length=255)
    call = models.CharField(max_length=50)
    work = models.CharField(max_length=100)
    url = models.CharField(max_length=255)
    category = models.CharField(max_length=20)

# 텍스트 테이블 1. 메일, 문자
class Text_mail(models.Model):
    id = models.AutoField(primary_key=True)
    filename = models.CharField(max_length=50, null=True)
    transcript = models.TextField()
    label = models.BooleanField() #default 설정 가능

# 텍스트 테이블 2. KorCCVi
class Text_KorCCVi(models.Model):
    id = models.AutoField(primary_key=True)
    transcript = models.TextField()
    label = models.BooleanField()

# 음성 파형
class Wave(models.Model):
    id = models.AutoField(primary_key=True)
    spectrogram = models.BigIntegerField()

# 번호 1. 전화번호
class Phone_numbers(models.Model):
    id = models.AutoField(primary_key=True)
    phone_number = models.CharField(max_length=50)
    count = models.IntegerField()

# 번호 2. 계좌번호
class Account_numbers(models.Model):
    id = models.AutoField(primary_key=True)
    account_number = models.CharField(max_length=50)
    count = models.IntegerField()


################
class Sentence(models.Model):
    text = models.CharField(max_length=255)
    embedding = models.TextField()