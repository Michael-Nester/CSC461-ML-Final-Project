import cv2
import numpy as np
import glob
import pickle
import os


if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("CUDA is available")
    USE_CUDA = True
else:
    print("CUDA is not available - falling back to CPU")
    USE_CUDA = False

# Create CUDA-enabled objects
eye_cascade_cuda = cv2.CascadeClassifier('../csc461/brn/haarcascade_eye.xml')
#gpu_stream = cv2.cuda.Stream()




# eye_cascade = cv2.CascadeClassifier('../csc461/brn/haarcascade_eye.xml')



def transform_image(img,threshold):

        if USE_CUDA:
            # Convert image to GPU
            gpu_img = cv2.cuda.GpuMat()
            gpu_img.upload(img)
        
            if threshold == 0:
                _, thresh = cv2.cuda.threshold(gpu_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, thresh = cv2.cuda.threshold(gpu_img, threshold, 255, cv2.THRESH_BINARY)
        
            # Create GPU kernel
            gpu_kernel = cv2.cuda.GpuMat()
            gpu_kernel.upload(kernel)
        
            # Perform morphological operations on GPU
            opening = cv2.cuda.morphologyEx(thresh, cv2.MORPH_OPEN, gpu_kernel)
            closing = cv2.cuda.morphologyEx(thresh, cv2.MORPH_CLOSE, gpu_kernel)
        
            # Download results back to CPU
            opening_cpu = opening.download()
            closing_cpu = closing.download()
        else:
            # CPU fallback
            if threshold == 0:
                _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
            
            opening_cpu = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            closing_cpu = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
        open_close = cv2.bitwise_or(opening_cpu, closing_cpu, mask=None)
        return open_close, opening_cpu, closing_cpu

imgs = []
label=0
final_output = []
lables = []
eye_detected_imgs=[]
iris_eye_detected_imgs=[]

path = '../csc461/brn/EYE_IMAGES_FULL'
#CORRECT WRITE         csv_path='../csc461/brn/iris_labels_full.csv',

print(path)

for filepath in glob.iglob(path):

    num_in_folder= 0

    for filefilepath in glob.iglob(filepath+'/*'):
      #print(filefilepath[-1])
      if filefilepath[-1] == 'f':
          #print(filefilepath[-1])
          img   = cv2.imread(filefilepath)
          imgs_colored=cv2.imread(filefilepath)
          img=cv2.resize(img,(200,150))

          img   =       cv2.cvtColor(img,       cv2.COLOR_BGR2GRAY)
          imgs.append([img,num_in_folder,label,imgs_colored])
          num_in_folder = num_in_folder+1
    label=label+1


eyes_num=0


for i,j,L,c in imgs:
    if USE_CUDA:
        # Convert to GPU
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(i)
        eyes = eye_cascade.detectMultiScale(i, 1.01, 0)
    else:
        eyes = eye_cascade.detectMultiScale(i, 1.01, 0)
    original_filename = os.path.basename(filefilepath)  # Get filename from filefilepath

    i=cv2.resize(i,(400,400))

    # Construct output path for final_iris images
    output_path_final_iris = os.path.join("../csc461/brn/final_iris/", original_filename)

    #eyes = eye_cascade.detectMultiScale(i, 1.01, 0)
    # Use CUDA-enabled eye detection
    eyes = eye_cascade_cuda.detectMultiScale(gpu_img).download()

    if len(eyes)>1:
        print(eyes_num)
        eye_detected_imgs.append(imgs[eyes_num])
        eyes_num = eyes_num+1



        maxium_area = -3

        for (ex,ey,ew,eh) in eyes:
            area = ew*eh

            if area>maxium_area:
                maxium_area = area
                maxium_width=ew
                point_x=ex
                point_y=ey
                maxium_height = eh

        #cv2.rectangle(c,(point_x,point_y),(point_x+maxium_width,+maxium_height),(255,0,0),2)

print("total_eyes_found = ",eyes_num)


print("total images number ",len(imgs))

iris_num=0
for i,j,L,c in eye_detected_imgs:


    circles = cv2.HoughCircles(i, cv2.HOUGH_GRADIENT, 10, 100)

    if circles is not None :

        circles = np.round(circles[0, :]).astype("int")
        #print(len(circles))
        #print(y)

        maxiumum_average=10000000000000
        #print(len(circles))
        #print(i.shape[0])
        #print(i.shape[1])
        #print(min(i.shape))

        key=True



        for (x, y, r) in circles:

            if x+r<=max(i.shape) and y+r<=max(i.shape)and x-r>0 and y-r>0 and r>20:

                key=False

                new_roi = i[y-r:y+r, x-r:x+r]
                average = np.average(new_roi)

                if average < maxiumum_average:
                    maxiumum_r = r
                    point_x=x
                    point_y=y
                    maxiumum_average=average


                #cv2.circle(i, (x, y), r, (0, 0, 0), 4)



        if key:
            #print("key opened")
            average = float('inf')
            for (x, y, r) in circles:


                    maxiumu_raduis=-4

                    if r > maxiumu_raduis:
                        maxiumum_r = r
                        point_x=x
                        point_y=y
                        maxiumum_average=average






        #cv2.circle(c, (point_x, point_y), maxiumum_r, (255, 255, 0), 4)
        #print(str(L)+'.'+str(j)+"  =  "+str(maxiumum_average)+"  "+str(r))

        #CORRECT WRITE cv2.imwrite("/content/drive/MyDrive/ML final project/datasets/iris/"+str(L)+'.'+str(j)+'.jpg',c)
        iris_eye_detected_imgs.append(eye_detected_imgs[iris_num])
        iris_num = iris_num+1


            #roi_gray = gray[y:y+h, x:x+w]
            #roi_gray = gray[ey:ey+eh, ex:ex+ew]
            #roi_color = img[ey:ey+eh, ex:ex+ew]


print("total_iris_found = ",iris_num)


print("total images number ",len(imgs))



imgs = iris_eye_detected_imgs




kernel = np.ones((5,5),np.uint8)
import random

random.shuffle(imgs)

test=[]
for i,j,L,c in imgs:

    gold,siver,diamond = transform_image(i,0)
    golden_refrence = sum(sum(gold))
    #print("golden refrence  = "+str(golden_refrence))



    for k in range(10,1000,10):

        #potential fix: dynamic threshold
        #_, working_img = cv2.threshold(i, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        working_img,opening,closing = transform_image(i,k)
        suming = sum(sum(working_img))
        diffrence = suming-golden_refrence

        if diffrence>800:
            print("the image threshold = " ,k)
            print("the image name " +str(j))
            print(" " )



            #CORRECT WRITE cv2.imwrite("/content/drive/MyDrive/ML final project/datasets/threshold/"+str(L)+'.'+str(j)+'.jpg',working_img)
            #CORRECT WRITE cv2.imwrite("/content/drive/MyDrive/ML final project/datasets/opening/"+str(L)+'.'+str(j)+'.jpg',opening)
            #CORRECT WRITE cv2.imwrite("/content/drive/MyDrive/ML final project/datasets/closing/"+str(L)+'.'+str(j)+'.jpg',closing)




            #_, contours,_ = cv2.findContours(working_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contours, _ = cv2.findContours(working_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            for z in contours:

                x,y,w,h = cv2.boundingRect(z)
                if x+w<150 and y+h<200 and x-w//4>0:

                    cv2.rectangle(working_img,(x,y),(x+w,y+h),(0,255,0),-2)
                    #CORRECT WRITE cv2.imwrite("/content/drive/MyDrive/ML final project/datasets/contour/"+str(L)+'.'+str(j)+'.jpg',working_img)


            contours_2,_ = cv2.findContours(working_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            #cv2.imwrite('paper/contour/'+str(L)+'.'+str(j)+'.jpg',contours_2)


            maxium_area=0
            maxium_area = 0
            maxium_width=0
            point_x=0
            point_y=0
            maxium_height = 0
            for z in contours_2:
                #print(len(i))
                x,y,w,h = cv2.boundingRect(z)
                new_area=h*w
                if x+w<150 and y+h<200 and new_area>maxium_area and x-w//4>0:
                    maxium_area = new_area
                    maxium_width=w
                    point_x=x
                    point_y=y
                    maxium_height = h


                    #cv2.rectangle(working_img,(x,y),(x+w,y+h),(0,255,0),-2)


            #cv2.rectangle(i,(point_x,point_y),(point_x+maxium_width,point_y+maxium_height),(0,255,0),-2)

            center_x = point_x+maxium_width//2
            center_y = point_y+maxium_height//2
            radius = 40

            if center_y-radius>0 and center_x-radius >0  and center_y+radius < 200 and center_x+radius < 150:
                #cv2.circle(c, (int(center_x), int(center_y)), int(radius),  (0, 255, 255), 2)
                new_roi = c[center_y-radius:center_y+radius, center_x-radius:center_x+radius]
                new_roi=cv2.resize(new_roi,(200,150))
                #new_roi        = cv2.cvtColor(new_roi,cv2.COLOR_GRAY2BGR)

                #cv2.imwrite('paper/threshold/'+str(L)+'.'+str(j)+'.jpg',new_roi)

                #cv2.imwrite("/content/drive/MyDrive/ML final project/datasets/final_iris/"+str(L)+'.'+str(j)+'.jpg',new_roi)
                cv2.imwrite(output_path_final_iris, new_roi)
            #new_roi=cv2.resize(new_roi,(200,150))
            else:
                center_y=c.shape[0]//2
                center_x=c.shape[1]//2
                new_roi = c[center_y-radius:center_y+radius, center_x-radius:center_x+radius]
                new_roi =cv2.resize(new_roi,(200,150))
                #new_roi = cv2.cvtColor(new_roi,cv2.COLOR_GRAY2BGR)

                cv2.imwrite("/content/drive/MyDrive/ML final project/datasets/final_iris/"+str(L)+'.'+str(j)+'.jpg',new_roi)

            #CORRECT WRITE cv2.imwrite("/content/drive/MyDrive/ML final project/datasets/edging_5/"+str(L)+'_'+str(j)+'.jpg',i)
            test.append(i)
            final_output.append(new_roi)
            lables.append(L)



            #cv2.imwrite('edging_5_test/'+str(j[5:]),i)

            break

print("the lenght of final output = ",len(final_output))
print("the of lables = ",len(lables))

final_output=np.array(final_output)
print(final_output.shape)

test=np.array(test)
print(test.shape)

pickle_out = open("test_ubiris.pickle","wb")
pickle.dump(test, pickle_out)
pickle_out.close()

pickle_out = open("ubiris_features.pickle","wb")
pickle.dump(final_output, pickle_out)
pickle_out.close()

pickle_out = open("ubiris_lables.pickle","wb")
pickle.dump(lables, pickle_out)
pickle_out.close()
