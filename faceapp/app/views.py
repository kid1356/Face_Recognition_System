from django.shortcuts import get_object_or_404, render, redirect
from django.http import HttpResponseForbidden, JsonResponse, StreamingHttpResponse
from .utils import *
from .forms import ImageUploadForm,UserForm
from .models import *
from django.contrib.auth import login, authenticate,logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt

# Create your views here.

@login_required
def recognize_page(request):
    return render(request , 'recognize.html')

@login_required
def stop_recognition(request):
    FaceRecognizer.stop_all()
    return JsonResponse({'status': 'stopped'})

@login_required
def face_recognition_view(request):
    # Create a new recognizer instance tied to this session
    recognizer = FaceRecognizer(request.session.session_key)
    
    def generator():
        recognizer.start_camera()
        try:
            while recognizer.running:
                frame = recognizer.get_frame()
                if frame is None:
                    break
                ret, jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        finally:
            recognizer.stop_camera()
    
    response = StreamingHttpResponse(generator(), content_type='multipart/x-mixed-replace; boundary=frame')
    return response

@login_required
def upload_image(request):
    if not request.user.is_superuser:
        return HttpResponseForbidden("Admin only")
    
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                recognizer = FaceRecognizer("upload_session")
                image_file = form.cleaned_data['image']
                person_name = form.cleaned_data['person_name'].strip()

                # Delete existing entries FOR CURRENT ADMIN
                Person.objects.filter(admin=request.user, name=person_name).delete()
                
                # Create new person WITH ADMIN
                person = Person.objects.create(
                    admin=request.user,
                    name=person_name
                )

                # Process image
                img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
                faces = recognizer.detect_faces(img)
                
                if len(faces) == 0:
                    raise ValueError("No faces detected")

                # Save all faces
                for box in faces:
                    x1, y1, x2, y2 = box
                    face_img = img[y1:y2, x1:x2]
                    
                    embedding = recognizer.process_face(face_img)
                    if embedding is not None:
                        FaceImage.objects.create(
                            person=person,
                            image=image_file,
                            embedding=embedding.tobytes()
                        )

                FaceRecognizer.retrain_global_model()
                return redirect('gallery')

            except Exception as e:
                Person.objects.filter(admin=request.user, name=person_name).delete()
                return render(request, 'error.html', {'error': str(e)})
    
    return render(request, 'upload.html', {'form': ImageUploadForm()})



@login_required
def gallery_view(request):
    images = FaceImage.objects.select_related('person__admin').all()
    return render(request, 'gallery.html', {'images': images})


@login_required
def home(request):
    return render(request, 'home.html')


def Delete_picture_view(request,image_id):
    if request.method == 'POST':
        image = get_object_or_404(FaceImage,id=image_id)
        image.delete()
        return JsonResponse({'success':True, 'redirect_url':'/gallery/'})
    return JsonResponse({"error": "Invalid request"}, status=400)  



def register_view(request):
    if request.method == 'POST':
        form = UserForm(request.POST)
        print(form)
        if form.is_valid():
            print('__________________')
            user = form.save()
            return redirect('login')
    else:
        form = UserForm()

    return render(request, 'register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def logoutview(request):

    logout(request)

    return render(request, 'logout.html')
