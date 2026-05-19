let currentCamera = null;

function qs(id) {
  return document.getElementById(id);
}

function normalizePlate(value) {
  return String(value || "")
    .toUpperCase()
    .replace(/\s+/g, "")
    .replace(/-/g, "")
    .replace(/_/g, "");
}

function toast(message, type = "info") {
  const old = document.querySelector(".toast");
  if (old) old.remove();

  const box = document.createElement("div");
  box.className = `toast ${type}`;
  box.textContent = message;

  document.body.appendChild(box);

  setTimeout(() => {
    box.classList.add("show");
  }, 20);

  setTimeout(() => {
    box.classList.remove("show");
    setTimeout(() => box.remove(), 250);
  }, 2600);
}

function openReason(camera) {
  currentCamera = camera;
  const modal = qs("reasonModal");
  if (modal) modal.style.display = "grid";
}

function closeModal() {
  const modal = qs("reasonModal");
  if (modal) modal.style.display = "none";
}

function confirmOpen() {
  const reasonSelect = qs("reasonSelect")?.value || "";
  const reasonText = qs("reasonText")?.value || "";
  const reason = `${reasonSelect}. ${reasonText}`.trim();

  if (!currentCamera) {
    toast("Камера таңдалмады", "error");
    return;
  }

  barrier(currentCamera, "open", reason);
  closeModal();
}

function barrier(name, action, reason = "") {
  const fd = new FormData();
  fd.append("name", name);
  fd.append("action", action);
  fd.append("reason", reason);

  fetch("/api/barrier", {
    method: "POST",
    body: fd,
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.ok) {
        const labels = {
          open: "Шлагбаум ашылды",
          close: "Шлагбаум жабылды",
          fix_open: "Ашық күйде фиксацияланды",
          fix_close: "Жабық күйде фиксацияланды",
          unfix: "Фиксация алынды",
        };

        toast(labels[action] || "Команда орындалды", "success");
      } else {
        toast(data.message || "Қате шықты", "error");
      }
    })
    .catch(() => toast("Серверге қосылу қатесі", "error"));
}

function manual(direction) {
  const inputId = direction === "entry" ? "manualEntry" : "manualExit";
  const input = qs(inputId);
  const plate = normalizePlate(input?.value);

  if (!plate) {
    toast("Номер жазыңыз", "error");
    return;
  }

  const fd = new FormData();
  fd.append("plate", plate);
  fd.append("direction", direction);

  fetch("/api/manual-entry", {
    method: "POST",
    body: fd,
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.ok) {
        toast("Журналға қосылды", "success");
        input.value = "";
        updateLastPlates();
      } else {
        toast(data.message || "Қате шықты", "error");
      }
    })
    .catch(() => toast("Серверге қосылу қатесі", "error"));
}

function updateLastPlates() {
  fetch("/api/last-plates")
    .then((r) => r.json())
    .then((data) => {
      const entry = qs("entryPlate");
      const exit = qs("exitPlate");

      if (entry) entry.textContent = data.entry || "---";
      if (exit) exit.textContent = data.exit || "---";
    })
    .catch(() => {});
}

function addPayment() {
  const plate = normalizePlate(qs("payPlate")?.value);
  const amount = Number(qs("payAmount")?.value || 0);
  const method = qs("payMethod")?.value || "Kaspi";

  if (!plate) {
    toast("Номер жазыңыз", "error");
    return;
  }

  if (amount <= 0) {
    toast("Соманы дұрыс жазыңыз", "error");
    return;
  }

  const fd = new FormData();
  fd.append("plate", plate);
  fd.append("amount", amount);
  fd.append("method", method);

  fetch("/api/payment", {
    method: "POST",
    body: fd,
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.ok) {
        toast("Төлем сақталды", "success");
        setTimeout(() => location.reload(), 700);
      } else {
        toast("Төлем сақталмады", "error");
      }
    })
    .catch(() => toast("Серверге қосылу қатесі", "error"));
}

function balance(operation) {
  const plate = normalizePlate(qs("balancePlate")?.value);
  const amount = Number(qs("balanceAmount")?.value || 0);

  if (!plate) {
    toast("Номер жазыңыз", "error");
    return;
  }

  if (amount <= 0) {
    toast("Соманы дұрыс жазыңыз", "error");
    return;
  }

  const fd = new FormData();
  fd.append("plate", plate);
  fd.append("amount", amount);
  fd.append("operation", operation);

  fetch("/api/balance", {
    method: "POST",
    body: fd,
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.ok) {
        toast(
          operation === "plus" ? "Баланс толықтырылды" : "Баланс азайтылды",
          "success",
        );
        setTimeout(() => location.reload(), 700);
      } else {
        toast("Баланс өзгермеді", "error");
      }
    })
    .catch(() => toast("Серверге қосылу қатесі", "error"));
}

function addList(type) {
  const plate = normalizePlate(qs("listPlate")?.value);
  const reason = qs("listReason")?.value || "";

  if (!plate) {
    toast("Номер жазыңыз", "error");
    return;
  }

  const fd = new FormData();
  fd.append("plate", plate);
  fd.append("type", type);
  fd.append("reason", reason);

  fetch("/api/list", {
    method: "POST",
    body: fd,
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.ok) {
        toast(
          type === "white" ? "Ақ списокқа қосылды" : "Қара списокқа қосылды",
          "success",
        );
        setTimeout(() => location.reload(), 700);
      } else {
        toast("Списокқа қосылмады", "error");
      }
    })
    .catch(() => toast("Серверге қосылу қатесі", "error"));
}

function selectPlate(plate) {
  const cleanPlate = normalizePlate(plate);

  const action = prompt(
    `${cleanPlate} үшін команда таңдаңыз:\n\n` +
      `white — ақ список\n` +
      `black — қара список\n` +
      `pay — төлем қосу`,
  );

  if (!action) return;

  if (action === "white" || action === "black") {
    const fd = new FormData();
    fd.append("plate", cleanPlate);
    fd.append("type", action);
    fd.append("reason", "Журналдан қосылды");

    fetch("/api/list", {
      method: "POST",
      body: fd,
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.ok) {
          toast(
            action === "white"
              ? "Ақ списокқа қосылды"
              : "Қара списокқа қосылды",
            "success",
          );
          setTimeout(() => location.reload(), 700);
        }
      });

    return;
  }

  if (action === "pay") {
    const amount = Number(prompt("Қанша теңге төледі?") || 0);

    if (amount <= 0) {
      toast("Сома дұрыс емес", "error");
      return;
    }

    const fd = new FormData();
    fd.append("plate", cleanPlate);
    fd.append("amount", amount);
    fd.append("method", "Kaspi");

    fetch("/api/payment", {
      method: "POST",
      body: fd,
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.ok) {
          toast("Төлем қосылды", "success");
          setTimeout(() => location.reload(), 700);
        }
      });
  }
}

/* CLIENT SLIDER */

function initClientSlider() {
  const slider = qs("slider");
  const plateInput = qs("clientPlate");
  const resultBox = qs("clientResult");

  if (!slider || !plateInput || !resultBox) return;

  let dragging = false;

  function resetSlider() {
    slider.style.left = "5px";
  }

  function checkStatus() {
    const plate = normalizePlate(plateInput.value);

    if (!plate) {
      toast("Номеріңізді жазыңыз", "error");
      resetSlider();
      return;
    }

    const fd = new FormData();
    fd.append("plate", plate);

    resultBox.innerHTML = `<p>Тексеріліп жатыр...</p>`;

    fetch("/api/client-status", {
      method: "POST",
      body: fd,
    })
      .then((r) => r.json())
      .then((data) => {
        resultBox.innerHTML = `
          <h3>${data.plate}</h3>
          ${data.messages.map((m) => `<p>• ${m}</p>`).join("")}
        `;
      })
      .catch(() => {
        resultBox.innerHTML = `<p>Серверге қосылу қатесі.</p>`;
      });

    resetSlider();
  }

  slider.addEventListener("mousedown", () => {
    dragging = true;
  });

  slider.addEventListener("touchstart", () => {
    dragging = true;
  });

  document.addEventListener("mouseup", () => {
    if (dragging) checkStatus();
    dragging = false;
  });

  document.addEventListener("touchend", () => {
    if (dragging) checkStatus();
    dragging = false;
  });

  document.addEventListener("mousemove", (event) => {
    if (!dragging) return;

    const track = document.querySelector(".slider-track");
    const rect = track.getBoundingClientRect();
    let x = event.clientX - rect.left - 30;

    if (x < 5) x = 5;
    if (x > rect.width - 65) x = rect.width - 65;

    slider.style.left = `${x}px`;
  });

  document.addEventListener("touchmove", (event) => {
    if (!dragging) return;

    const touch = event.touches[0];
    const track = document.querySelector(".slider-track");
    const rect = track.getBoundingClientRect();
    let x = touch.clientX - rect.left - 30;

    if (x < 5) x = 5;
    if (x > rect.width - 65) x = rect.width - 65;

    slider.style.left = `${x}px`;
  });
}

document.addEventListener("DOMContentLoaded", () => {
  updateLastPlates();
  initClientSlider();

  setInterval(updateLastPlates, 1500);
});

document.addEventListener("DOMContentLoaded", function () {
  const currentPath = window.location.pathname;
  const links = document.querySelectorAll(".nav-link");

  links.forEach((link) => {
    link.classList.remove("active");

    const href = link.getAttribute("href");

    if (href === currentPath) {
      link.classList.add("active");
    }
  });
});

function addAbonement() {
  const fullName = qs("abonFullName")?.value.trim();
  const phone = qs("abonPhone")?.value.trim();
  const plate = normalizePlate(qs("abonPlate")?.value);
  const groupName = qs("abonGroup")?.value.trim();

  if (!fullName || !phone || !plate) {
    toast("Аты-жөні, телефон және машина номері міндетті", "error");
    return;
  }

  const fd = new FormData();
  fd.append("full_name", fullName);
  fd.append("phone", phone);
  fd.append("plate", plate);
  fd.append("group_name", groupName || "");

  fetch("/api/abonement", {
    method: "POST",
    body: fd,
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.ok) {
        toast("Абонемент тіркелді", "success");
        setTimeout(() => location.reload(), 700);
      } else {
        toast(data.message || "Абонемент тіркелмеді", "error");
      }
    })
    .catch(() => toast("Серверге қосылу қатесі", "error"));
}

function addAdministration() {
  const fullName = qs("adminFullName")?.value.trim();
  const phone = qs("adminPhone")?.value.trim();
  const plate = normalizePlate(qs("adminPlate")?.value);
  const position = qs("adminPosition")?.value.trim();

  if (!fullName || !phone || !plate || !position) {
    toast("Барлық мәліметті толтырыңыз", "error");
    return;
  }

  const fd = new FormData();
  fd.append("full_name", fullName);
  fd.append("phone", phone);
  fd.append("plate", plate);
  fd.append("position", position);

  fetch("/api/administration", {
    method: "POST",
    body: fd,
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.ok) {
        toast("Қызметкер тіркелді", "success");
        setTimeout(() => location.reload(), 700);
      } else {
        toast(data.message || "Қызметкер тіркелмеді", "error");
      }
    })
    .catch(() => toast("Серверге қосылу қатесі", "error"));
}

function disableAccess(table, plate) {
  if (!confirm(`${plate} рұқсатын өшіреміз бе?`)) return;

  const fd = new FormData();
  fd.append("table", table);
  fd.append("plate", normalizePlate(plate));

  fetch("/api/free-access/delete", {
    method: "POST",
    body: fd,
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.ok) {
        toast("Рұқсат өшірілді", "success");
        setTimeout(() => location.reload(), 700);
      } else {
        toast("Рұқсат өшірілмеді", "error");
      }
    })
    .catch(() => toast("Серверге қосылу қатесі", "error"));
}

function payAbonement(plate) {
  if (!confirm(`${plate} абонемент төлемін растаймыз ба?`)) return;

  const fd = new FormData();
  fd.append("plate", normalizePlate(plate));

  fetch("/api/abonement/pay", {
    method: "POST",
    body: fd,
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.ok) {
        toast("Абонемент төленді", "success");
        setTimeout(() => location.reload(), 700);
      } else {
        toast(data.message || "Қате шықты", "error");
      }
    })
    .catch(() => toast("Серверге қосылу қатесі", "error"));
}
