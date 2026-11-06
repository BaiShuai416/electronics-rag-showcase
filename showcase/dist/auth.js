"use strict";

const params = new URLSearchParams(location.search);
const target = ["/recipes", "/devices", "/admin"].includes(params.get("next")) ? params.get("next") : "/recipes";
const isLogin = location.pathname.replace(/\/$/, "") === "/login";
const isAdminLogin = isLogin && target === "/admin";
const form = document.getElementById("auth-form");
const errorBox = document.getElementById("auth-error");
const submit = document.getElementById("auth-submit");
const nameField = document.getElementById("name-field");
const loginTab = document.getElementById("login-tab");
const registerTab = document.getElementById("register-tab");

loginTab.href = "/login?next=" + encodeURIComponent(target);
registerTab.href = "/register?next=" + encodeURIComponent(target === "/admin" ? "/recipes" : target);
if (isLogin) {
  document.getElementById("auth-title").textContent = isAdminLogin ? "管理员登录" : "登录访客账号";
  if (isAdminLogin) {
    document.getElementById("auth-intro").textContent = "使用你在本机创建的管理员账号。";
    document.querySelector(".auth-tabs").hidden = true;
  }
  nameField.hidden = true;
  form.elements.display_name.disabled = true;
  form.elements.password.autocomplete = "current-password";
  loginTab.setAttribute("aria-current", "page");
  registerTab.removeAttribute("aria-current");
  submit.textContent = "登录 ↗";
  form.action = "/api/auth/login";
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  submit.disabled = true;
  errorBox.hidden = true;
  try {
    const response = await fetch(form.action, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(Object.fromEntries(new FormData(form))),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "提交失败，请稍后重试。");
    if (isAdminLogin && !data.is_admin) {
      await fetch("/api/auth/logout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      throw new Error("请使用管理员账号登录。");
    }
    location.replace(target === "/admin" && !data.is_admin ? "/recipes" : target);
  } catch (error) {
    errorBox.textContent = error.message || "连接失败，请稍后重试。";
    errorBox.hidden = false;
    submit.disabled = false;
  }
});
form.hidden = false;
