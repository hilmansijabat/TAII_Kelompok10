let dataString = localStorage.getItem("last-result");
let data = JSON.parse(dataString).data;
console.log(data);
console.log("interval :" + data.interval);
let inputPrice = document.getElementById("price");
let inputVolume = document.getElementById("interval");

inputPrice.value = data.price;
inputVolume.value = data.interval;
// inputVolume.value = data.interval.toFixed(2) + " mL";
