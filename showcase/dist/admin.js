"use strict";

const pages = { users: 1, questions: 1 };
const groups = [...document.querySelectorAll("[data-page-group]")];
const refresh = document.getElementById("refresh-records");
const logout = document.getElementById("admin-logout");
const errorBox = document.getElementById("admin-error");
const dateTime = new Intl.DateTimeFormat("zh-CN", { dateStyle: "medium", timeStyle: "medium" });
const statuses = { answered: "已完成", pending: "待处理", busy: "服务忙", failed: "未完成" };

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function fillTable(id, rows, width, emptyMessage) {
  const table = document.getElementById(id);
  table.replaceChildren();
  if (!rows.length) {
    const cell = table.insertRow().insertCell();
    cell.colSpan = width;
    cell.textContent = emptyMessage;
    return;
  }
  for (const values of rows) {
    const row = table.insertRow();
    for (const value of values) row.insertCell().textContent = value;
  }
}

async function loadRecords(nextPages = pages) {
  refresh.disabled = true;
  errorBox.hidden = true;
  const buttons = groups.flatMap(group => [...group.querySelectorAll("button")]);
  const disabled = buttons.map(button => button.disabled);
  buttons.forEach(button => { button.disabled = true; });
  try {
    const response = await fetch("/api/admin/overview?" + new URLSearchParams(nextPages));
    if (response.status === 401) {
      location.replace("/login?next=%2Fadmin");
      return;
    }
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "读取记录失败。");
    Object.assign(pages, nextPages);
    document.getElementById("user-count").textContent = data.user_count;
    document.getElementById("question-count").textContent = data.question_count;
    fillTable("users-body", data.users.map(user => [
      user.username, user.display_name, user.is_admin ? "管理员" : "访客",
      dateTime.format(new Date(user.created_at)),
    ]), 4, "暂无账号。");
    fillTable("questions-body", data.questions.map(entry => [
      entry.display_name + "（" + entry.username + "）",
      entry.project === "recipes" ? "菜谱问答" : "器件手册", entry.question,
      (statuses[entry.status] || "未知") + (entry.route ? " · " + entry.route : ""),
      dateTime.format(new Date(entry.created_at)),
    ]), 5, "暂无提问。");
    for (const group of groups) {
      const key = group.dataset.pageGroup;
      const count = key === "users" ? data.user_count : data.question_count;
      const last = Math.max(1, Math.ceil(count / data.page_size));
      group.querySelector("[data-page-label]").textContent = "第 " + pages[key] + " / " + last + " 页";
      group.querySelector('[data-step="-1"]').disabled = pages[key] <= 1;
      group.querySelector('[data-step="1"]').disabled = pages[key] >= last;
    }
  } catch (error) {
    showError(error.message || "读取记录失败，请刷新重试。");
    buttons.forEach((button, index) => { button.disabled = disabled[index]; });
  } finally {
    refresh.disabled = false;
  }
}

for (const group of groups) {
  const key = group.dataset.pageGroup;
  for (const button of group.querySelectorAll("button")) {
    button.addEventListener("click", () => loadRecords({ ...pages, [key]: pages[key] + Number(button.dataset.step) }));
  }
}
refresh.addEventListener("click", () => loadRecords({ users: 1, questions: 1 }));
logout.addEventListener("click", async () => {
  logout.disabled = true;
  try {
    const response = await fetch("/api/auth/logout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (!response.ok) throw new Error();
    location.replace("/login?next=%2Fadmin");
  } catch {
    showError("退出失败，请稍后重试。");
    logout.disabled = false;
  }
});
loadRecords();
