import math

from django.shortcuts import render

from django.conf import settings
from rembg import remove
from PIL import Image
from django.http import JsonResponse
import os
import numpy as np
import cv2 as cv
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from fastai.vision import *
from fastai.metrics import error_rate, accuracy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import zipfile
from django.http import JsonResponse
import matplotlib.pyplot as plt
import time
import os
from django import template
from fastai.vision.data import ImageDataLoaders
from fastai.vision.all import *

pixelsPerMetric = 46
pixelsPerMetricT = 43

# TODO: Create view for index


def index(request):
    context = {}
    return render(request, 'scanning/index.html', context)


def scanning_request(request):
    context = {}
    return render(request, 'scanning/scanning.html', context)


def detail(request):
    context = {}
    return render(request, 'scanning/detail.html', context)


def scanning_process(request):
    data = {}
    response = {}
    if request.method == 'POST':
        count = 1
        width = 0
        height = 0
        length = 0
        for key in request.FILES:
            file = request.FILES[key]
            name = file.name
            splitter = name.split("-")
            input = Image.open(file)
            output = remove(input)
            path = os.path.abspath(settings.BASE_DIR) + settings.STATIC_URL + "temp/file-" + splitter[0] + "-" + str(
                count) + ".png"
            path_save = os.path.abspath(settings.BASE_DIR) + settings.STATIC_URL + "temp_count/file-" + splitter[
                0] + "-" + str(
                count) + ".png"
            output.save(path)
            size_a, size_b = start_count_width_height(path, path_save)
            if count == 1:
                height = size_b
                if size_a > size_b:
                    height = size_a
            else:
                width = size_b
                length = size_a
            data["file_" + str(count)] = {
                "access_file": settings.STATIC_URL + "temp/file-" + splitter[0] + "-" + str(count) + ".png",
                "count_file": settings.STATIC_URL + "temp_count/file-" + splitter[0] + "-" + str(count) + ".png",
            }
            count += 1
        used_radius = width
        if length > width:
            used_radius = length
        # volume = math.pi * math.pow((used_radius / 2), 2) * height
        area = width * height
        volume = calculate_bottle_size_ml(area)
        data["size"] = {
            "width": width,
            "length": length,
            "height": height,
            "volume": volume,
        }
        volume = data["size"]["volume"]
        data["interval"] = get_bottle_size_range(volume)

        price = (volume / 1000) * 200
        data["price"] = f'Rp.{price:,.2f}'
    response = {
        "data": data
    }
    return JsonResponse(response)


def calculate_bottle_size_ml(area):
    if area <= 123:
        return 282
    elif 123 < area <= 189:
        return 600
    elif 189 < area <= 217:
        return 912.5
    elif area > 217:
        return 1500


def get_bottle_size_range(volume):
    # Add your logic to determine the range based on the volume
    if volume <= 282:
        return "175ml - 390ml"
    elif volume <= 600:
        return "450ml - 750ml"
    elif volume <= 912.5:
        return "825ml - 1000ml"
    else:
        return "1500ml or more"


def start_count_width_height(path, path_save):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    ret, thresh = cv.threshold(img, 127, 255, 0)
    cnts, hierarchy = cv.findContours(thresh, 1, 2)
    temp = [c for c in cnts if cv.contourArea(c) > 100]

    biggest = temp[0]

    for t in temp:
        if cv.contourArea(t) > cv.contourArea(biggest):
            biggest = t

    c = biggest

    # Mempertahankan gambar asli dan mencegah adanya perubahan pada gambar asli
    orig = img.copy()
    box = cv.minAreaRect(c)
    box = cv.cv.BoxPoints(box) if imutils.is_cv2() else cv.boxPoints(box)
    box = np.array(box, dtype="int")

    # proses gambar
    box = perspective.order_points(box)
    cv.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    for (x, y) in box:
        cv.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # menggambar titik tengah
    cv.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # menggambar garis tengah
    cv.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
            (255, 0, 255), 4)
    cv.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
            (255, 0, 255), 4)

    # menghitung jarak antar titik tengah
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # menghitung ukuran panjang dan lebar
    dimA = dA / pixelsPerMetricT
    dimB = dB / pixelsPerMetric

    # menggambar ukuran panjang dan lebar
    cv.putText(orig, "{:.1f}in".format(dimA),
               (int(tltrX - 15), int(tltrY - 10)), cv.FONT_HERSHEY_SIMPLEX,
               0.65, (255, 255, 255), 2)
    cv.putText(orig, "{:.1f}in".format(dimB),
               (int(trbrX + 10), int(trbrY)), cv.FONT_HERSHEY_SIMPLEX,
               0.65, (255, 255, 255), 2)

    # menampilkan gambar
    cv.imwrite(path_save, orig)
    size_a = 2.54 * dimA  # cm
    size_b = 2.54 * dimB  # cm

    return size_a, size_b


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


def handle_uploaded_file(f):
    with open("some/file/name.txt", "wb+") as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def scanning_image(request):
    response = {}
    if request.method == 'POST':
        file = request.FILES["image_1"]
        path = os.path.abspath(settings.BASE_DIR) + \
            settings.STATIC_URL + "model/export3.pkl"
        loaded_model = load_learner(path)
        predict = loaded_model.predict(file.read())[0]
        response["data"] = predict
    return JsonResponse(response)
