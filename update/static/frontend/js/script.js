const fileInput = document.getElementById("customFile");
const customTxt = document.getElementById("choose-file");
const submitButton = document.getElementById("start");
const loading = document.getElementById("loadAnim");

// $(".loadAnim").css("display", "none");

fileInput.addEventListener("change", function () {
    
    if (fileInput.value) {
        customTxt.innerHTML = fileInput.files.length.toString() + " Files choosen";       
    }
})

// const fileInput = document.querySelector('#file-input');

submitButton.addEventListener('click', async (e) => {
  if (fileInput.value) {
    loading.style.display = "block"
    submitButton.innerHTML = "Scanning...";  
    e.preventDefault();
    const formData = new FormData();
    const fileName = fileInput.files[0].name
    // formData.append('file', fileInput.files[0], fileInput.files[0].name);
    for (let i = 0; i < fileInput.files.length; i++) {
      formData.append('files', fileInput.files[i]);
    }
    debugger;
    console.log(fileInput.files[0].name);
    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });

    const res = await response.json();
    console.log(res);
    
    let result = false
    result = await fetch('/extract');
    loading.style.display = "none"
    customTxt.innerHTML = "Drop files here to upload"; 
    submitButton.innerHTML = "Upload & Scan"; 
    // if (result){
    //     window.open(`/result/${fileName}`, "_blank")   
    // }
  }
});