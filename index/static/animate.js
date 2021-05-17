$(document).ready(function () {
    $(document).on("scroll", onScroll);
    //smoothscroll
    $('.navbar-nav li a[href^="#"]').on('click', function (e) {
        e.preventDefault();
        $(document).off("scroll");
        $('.navbar-nav li a').each(function () {
            $(this).removeClass('active');
        })
        $(this).addClass('active');
        var target = this.hash
            , menu = target;
        target = $(target);
        $('html, body').stop().animate({
            'scrollTop': target.offset().top + 2
        }, 1000, 'swing', function () {
            window.location.hash = target;
            $(document).on("scroll", onScroll);
        });
    });
});

function onScroll(event) {
    var scrollPos = $(document).scrollTop();
    $('.navbar-nav li a').each(function () {
        var currLink = $(this);
        var refElement = $(currLink.attr("href"));
        if (refElement.position().top <= scrollPos && refElement.position().top + refElement.height() > scrollPos) {
            $('.navbar-nav li a').removeClass("active");
            currLink.addClass("active");
        }
        else {
            currLink.removeClass("active");
        }
    });
}


const fileImage = document.querySelector('.input-preview__src');
const filePreview = document.querySelector('.input-preview');

fileImage.onchange = function () {
    const reader = new FileReader();

    reader.onload = function (e) {
        // get loaded data and render thumbnail.
        filePreview.style.backgroundImage  = "url("+e.target.result+")";
        filePreview.classList.add("has-image");
    };

    // read the image file as a data URL.
    reader.readAsDataURL(this.files[0]);
};