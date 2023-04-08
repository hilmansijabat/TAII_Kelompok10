let video = document.querySelector("#video-webcam");
let btnTakePicture = document.querySelector("#btn-start");
let textNumberCount = document.querySelector("#number-count");
let textNext = document.querySelector("#text-next");
let result = document.querySelector("#result");
// minta izin user
navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia ||
    navigator.mozGetUserMedia ||
    navigator.msGetUserMedia ||
    navigator.oGetUserMedia;
// jika user memberikan izin
if (navigator.getUserMedia) {
    // jalankan fungsi handleVideo, dan videoError jika izin ditolak
    navigator.getUserMedia({video: true}, handleVideo, videoError);
}

function getMachineId() {

    let machineId = localStorage.getItem('MachineId');

    if (!machineId) {
        machineId = crypto.randomUUID();
        localStorage.setItem('MachineId', machineId);
    }

    return machineId;
}

// fungsi ini akan dieksekusi jika  izin telah diberikan
function handleVideo(stream) {
    video.srcObject = stream;
}

// fungsi ini akan dieksekusi kalau user menolak izin
function videoError(e) {
    // do something
    alert("Izinkan menggunakan webcam untuk demo!")
}

let counterTime = 5;
let counterCount = 0;
let maxTakePicture = 2;
let takePicture = 0;
let timer;
let files;
let form = new FormData();

function callTimer() {
    timer = setInterval(() => {
        counterCount++;
        textNumberCount.classList.remove("hide");
        textNumberCount.innerHTML = `${counterCount}`;
        if (counterCount === counterTime) {
            console.log(`Finisih ${counterCount}`);
            counterCount = 0;
            takePicture += 1;
            clearInterval(timer);
            textNumberCount.classList.add("hide");
            textNext.classList.remove("hide");
            textNext.innerHTML = `We will take a picture`;
            setTimeout(() => {
                textNext.classList.add("hide");
                // buat elemen img
                var img = document.createElement('img');
                var context;

                // ambil ukuran video
                var width = video.offsetWidth
                    , height = video.offsetHeight;

                // buat elemen canvas
                canvas = document.createElement('canvas');
                canvas.width = width;
                canvas.height = height;

                // ambil gambar dari video dan masukan
                // ke dalam canvas
                context = canvas.getContext('2d');
                context.drawImage(video, 0, 0, width, height);

                canvas.toBlob(function (blob) {

                    console.log(typeof (blob)) //let you have 'blob' here
                    let fileOfBlob = new File([blob], `${getMachineId()}-${takePicture}.png`);
                    form.append(`image_${takePicture}`, fileOfBlob);
                    textNext.classList.remove("hide");
                        textNext.innerHTML = `We done take the picture and compute the size it..`;
                    if (takePicture === maxTakePicture) {
                        axios.defaults.xsrfCookieName = 'csrftoken'
                        axios.defaults.xsrfHeaderName = "X-CSRFTOKEN"
                        axios.post("/scanning/process", form).then(function (response) {
                            console.log(response)
                            localStorage.setItem("last-result", JSON.stringify(response.data));
                            window.location.href = "/scanning/detail/";
                            // do whatever you want if console is [object object] then stringify the response
                        })
                    } else if (takePicture === 1) {
                        textNext.classList.remove("hide");
                        textNext.innerHTML = `We done take the picture and identification it..`;
                        axios.defaults.xsrfCookieName = 'csrftoken'
                        axios.defaults.xsrfHeaderName = "X-CSRFTOKEN"
                        axios.post("/scanning/predict", form).then(function (response) {
                            console.log(response)
                            // do whatever you want if console is [object object] then stringify the response
                            if (response.data.data !== "bottle_plastic") {
                                result.innerHTML = "";
                                counterCount = 0;
                                takePicture = 0;
                                form = new FormData();
                                textNext.innerHTML = "It's not bottle plastic"
                                btnTakePicture.innerHTML = "Try Again"
                                btnTakePicture.classList.remove("hide")
                                textNumberCount.classList.add("hide")
                            } else if (takePicture < maxTakePicture) {
                                textNext.classList.remove("hide");
                                textNext.innerHTML = `We will take the next picture.`;
                                setTimeout(() => {
                                    textNext.classList.add("hide");
                                    callTimer();
                                }, 2000);
                            } else {
                                textNext.classList.remove("hide");
                                textNext.innerHTML = `We done take the picture and process it..`;
                            }
                        })
                    }
                    // render hasil dari canvas ke elemen img
                    img.src = canvas.toDataURL('image/png');
                    result.appendChild(img);

                }, 'image/png', 1);
            }, 1000)
        }
    }, 1000);
}

btnTakePicture.addEventListener("click", function (event) {
    btnTakePicture.classList.add("hide")
    callTimer();
});