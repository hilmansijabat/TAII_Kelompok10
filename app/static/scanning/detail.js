let dataString = localStorage.getItem("last-result");
let data = JSON.parse(dataString).data
console.log(data);
console.log(data.size);
let inputPrice = document.getElementById("price");
let inputVolume = document.getElementById("volume");

inputPrice.value = data.price;
inputVolume.value = data.size.volume.toFixed(2) + " mL";