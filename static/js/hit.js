window.onload = function () {



}


function hello() {
    var msg = document.getElementById("email-6564").value
    // var textarea = document.getElementById('emai6564');
    $.LoadingOverlay("show");
    axios.post('http://139.150.65.139:5000/mbti/result', {
        msg: msg
    })
        .then(function (response) {
            const checkMsg = response.data
            alert(checkMsg, 'info')
            console.log(checkMsg);

            $.LoadingOverlay("hide");



            var imageUrl = "http://139.150.65.139:5000/popup?mbti=" + checkMsg;
            window.open(imageUrl, 'popupWindow', 'width=400,height=500')


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




var progress = {
    color: '#1976d2'
};
progress.hide = function () {
    document.getElementById("progress-a-dim").remove();
}
progress.show = function () {
    if (document.getElementById("progress-a-dim") == null) {
        var style = document.createElement('style');
        var html = `      
        #progress-a-dim{position:fixed;left:0px;top:0px;width:100vw;height:100vh;display:flex;align-items:center;justify-content:center;z-index:9999999999999999999999}.progress-a{width:40px;height:40px;display:inline-block;color:${progress.color};animation:progress-a-1 1.4s linear infinite}.progress-b{stroke:#00AB97;stroke-dasharray:80px,200px;stroke-dashoffset:0;-webkit-animation:progress-a-2 1.4s ease-in-out infinite;animation:progress-a-2 1.4s ease-in-out infinite}@-webkit-keyframes progress-a-1{0%{-webkit-transform:rotate(0);-moz-transform:rotate(0);-ms-transform:rotate(0);transform:rotate(0)}100%{-webkit-transform:rotate(360deg);-moz-transform:rotate(360deg);-ms-transform:rotate(360deg);transform:rotate(360deg)}}@keyframes progress-a-1{0%{-webkit-transform:rotate(0);-moz-transform:rotate(0);-ms-transform:rotate(0);transform:rotate(0)}100%{-webkit-transform:rotate(360deg);-moz-transform:rotate(360deg);-ms-transform:rotate(360deg);transform:rotate(360deg)}}@-webkit-keyframes progress-a-2{0%{stroke-dasharray:1px,200px;stroke-dashoffset:0}50%{stroke-dasharray:100px,200px;stroke-dashoffset:-15px}100%{stroke-dasharray:100px,200px;stroke-dashoffset:-125px}}@keyframes progress-a-2{0%{stroke-dasharray:1px,200px;stroke-dashoffset:0}50%{stroke-dasharray:100px,200px;stroke-dashoffset:-15px}100%{stroke-dasharray:100px,200px;stroke-dashoffset:-125px}}      
      `;
        style.id = 'progress-a-style';
        style.innerHTML = (html);
        document.body.append(style);
    }
    if (document.getElementById("progress-a-dim") == null) {
        var div = document.createElement('div');

        var html = `
        <span class="progress-a" role="progressbar"><svg viewBox="22 22 44 44"><circle class="progress-b" cx="44" cy="44" r="20.2" fill="none" stroke-width="3.6"></circle></svg></span>      
      `;
        div.id = "progress-a-dim";
        div.innerHTML = (html);
        document.body.append(div);
    }
}