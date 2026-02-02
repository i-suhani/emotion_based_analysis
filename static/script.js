function startCamera() {
    document.getElementById("video").src = "/video_feed";
    setInterval(fetchData, 1000);
}

async function fetchData() {
    const res = await fetch("/detect");
    const data = await res.json();

    document.getElementById("smile").innerText = data.smile;
    document.getElementById("emotion").innerText = data.emotion;
    document.getElementById("age").innerText = data.age;
    document.getElementById("mask").innerText = data.mask;

    const extra = document.getElementById("extra");
    extra.innerHTML = "";

    if (data.emotion === "Sad") {
        extra.innerHTML = "😄 Joke: Why don’t scientists trust atoms? Because they make up everything!";
    }

    if (data.emotion === "Angry") {
        extra.innerHTML = `<iframe width="300" height="80"
            src="https://www.youtube.com/embed/2OEL4P1Rz04"></iframe>`;
    }

    if (data.emotion === "Fear") {
        extra.innerHTML = "⚠ ALERT SENT TO GUARDIAN!";
    }
}
