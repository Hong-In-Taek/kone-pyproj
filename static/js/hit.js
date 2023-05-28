window.onload = function () {



}


function hello() {
    var msg = document.getElementById("email-6564").value
    // var textarea = document.getElementById('emai6564');

    axios.post('http://139.150.65.139:5000/mbti/result', {
        msg: msg
    })
        .then(function (response) {
            const checkMsg = response.data
            alert(checkMsg, 'info')
            console.log(checkMsg);


            var imageUrl = response.image_url;
            showPopup(imageUrl);


            //if (checkMsgArray[0] == 1){
            //    alert(checkMsgArray[1], 'warning')
            //}else{
            //    alert(checkMsgArray[1], 'success')
            //}

        })
        .catch(function (error) {
            console.log(error);
        });
}


function training() {
    var msg = document.getElementById("email-6564").value
    // var textarea = document.getElementById('emai6564');

    axios.get('http://139.150.65.139:5000/mbti/train')
        .then(function (response) {
            alert("학습이 완료되었습니다.", 'info')
        })
        .catch(function (error) {
            console.log(error);
        });
}





var alert = function (msg, type) {
    swal({
        title: '',
        text: msg,
        type: type,
        timer: 1500,
        customClass: 'sweet-size',
        showConfirmButton: false
    });
}


function showPopup(imageUrl) {
    var popup = document.getElementById('popup');
    var image = document.getElementById('popup-image');
    image.src = imageUrl;
    popup.style.display = 'block';
}

function hidePopup() {
    var popup = document.getElementById('popup');
    popup.style.display = 'none';
}