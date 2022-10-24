window.onload = function () {



}


function hello() {
    var msg = document.getElementById("email-6564").value
    // var textarea = document.getElementById('emai6564');
    alert("sss", 'success')
    axios.post('http://localhost:5000/test', {
        msg: msg
    })
        .then(function (response) {
            alert(response.data, 'success')
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