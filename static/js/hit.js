window.onload = function () {



}


function hello() {
    var msg = document.getElementById("email-6564").value
    // var textarea = document.getElementById('emai6564');

    axios.post('http://localhost:5000/result', {
        msg: msg
    })
        .then(function (response) {
            const checkMsg  = response.data
            checkMsgArray = checkMsg.split("#")
            if (checkMsgArray[0] == 1){
                alert(checkMsgArray[1], 'warning')
            }else{
                alert(checkMsgArray[1], 'success')
            }
            
        })
        .catch(function (error) {
            console.log(error);
        });
}


function training() {
    var msg = document.getElementById("email-6564").value
    // var textarea = document.getElementById('emai6564');

    axios.get('http://localhost:5000/train')
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