"use strict";

const demo = document.body.dataset.demo;
const form = document.getElementById("ask-form");
const input = document.getElementById("question");
const result = document.getElementById("result");
const availability = document.querySelector("[data-availability]");
const viewer = document.querySelector("[data-viewer]");
const signOut = document.querySelector("[data-sign-out]");
const submit = form.querySelector("[type=submit]");
const modeButtons = Array.from(document.querySelectorAll("[data-mode]"));
let mode = "preview";
let selectedModel = "DS18B20";
let canGenerate = false;
let busy = false;
let recipes = [];
let recipeOffset = 0;

function returnToLogin() {
  location.replace("/login?next=" + encodeURIComponent("/" + demo));
}

signOut.addEventListener("click", async () => {
  signOut.disabled = true;
  try {
    const response = await fetch("/api/auth/logout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (!response.ok) throw new Error();
    returnToLogin();
  } catch {
    result.hidden = false;
    result.replaceChildren(element("p", "退出失败，请稍后重试。", "error"));
    signOut.disabled = false;
  }
});

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function showAvailability(remaining) {
  availability.classList.add("ready");
  availability.replaceChildren(element("span"), document.createTextNode(
    canGenerate && remaining > 0 ? "演示在线" : "证据预览"
  ));
  modeButtons.find(button => button.dataset.mode === "answer").disabled = !canGenerate || remaining === 0;
}

function selectMode(next) {
  mode = next;
  modeButtons.forEach(button => button.setAttribute("aria-pressed", String(button.dataset.mode === mode)));
  submit.textContent = mode === "answer" ? "生成回答 ↑" : (demo === "recipes" ? "检索菜谱 ↑" : "查询证据 ↑");
}
modeButtons.forEach(button => button.addEventListener("click", () => selectMode(button.dataset.mode)));

function selectModel(model) {
  selectedModel = model;
  document.querySelectorAll("[data-model]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.model === model)));
  const chip = document.querySelector("[data-chip-model]");
  if (chip) chip.textContent = model;
  input.placeholder = "例如：" + model + " 在 90°C 的温度误差是多少？";
}
document.querySelectorAll("[data-model]").forEach(button => button.addEventListener("click", () => selectModel(button.dataset.model)));

function askExample(question) {
  if (busy) return;
  input.value = question;
  document.getElementById("ask").scrollIntoView({ block: "start" });
  form.requestSubmit();
}
document.querySelectorAll("[data-question], [data-template]").forEach(button => {
  button.addEventListener("click", () => askExample(button.dataset.question || button.dataset.template.replace("{model}", selectedModel)));
});

function renderRecipeCards() {
  const grid = document.getElementById("recipe-cards");
  if (!grid || recipes.length === 0) return;
  grid.replaceChildren();
  for (let i = 0; i < Math.min(3, recipes.length); i += 1) {
    const recipe = recipes[(recipeOffset + i) % recipes.length];
    const button = element("button", null, "recipe-card");
    button.type = "button";
    button.setAttribute("aria-label", "查看" + recipe.name + "的做法");
    const image = element("img");
    image.src = recipe.image;
    image.alt = recipe.name;
    image.loading = "lazy";
    const copy = element("span", null, "recipe-card-copy");
    const name = element("span");
    name.append(element("small", recipe.category), element("strong", recipe.name));
    copy.append(name, element("span", "↗", "card-arrow"));
    button.append(image, copy);
    button.addEventListener("click", () => askExample(recipe.name + "怎么做？"));
    grid.append(button);
  }
}
const shuffle = document.getElementById("shuffle");
if (shuffle) shuffle.addEventListener("click", () => {
  if (recipes.length === 0) return;
  recipeOffset = (recipeOffset + 3) % recipes.length;
  renderRecipeCards();
});

function appendInline(node, text, sources) {
  // User and model output stays as text. Only evidence URLs supplied by the server become links.
  const tokens = text.split(/(\*\*[^*]+\*\*|【[^】]+】)/g);
  for (const token of tokens) {
    if (token.startsWith("**") && token.endsWith("**") && token.length > 4) {
      node.append(element("strong", token.slice(2, -2)));
      continue;
    }
    const evidence = sources.find(source => "【" + source.location + "】" === token && source.href);
    if (evidence) {
      const link = element("a", token);
      link.href = evidence.href;
      link.target = "_blank";
      link.rel = "noopener";
      node.append(link);
    } else {
      node.append(document.createTextNode(token));
    }
  }
}

function renderAnswer(answer, sources) {
  const body = element("div", null, "answer-body");
  let list = null;
  for (const raw of String(answer || "").split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) {
      list = null;
      continue;
    }
    const bullet = line.match(/^(?:[-*]\s+|(\d+)\.\s+)(.+)$/);
    if (bullet) {
      const tag = bullet[1] ? "OL" : "UL";
      if (!list || list.tagName !== tag) {
        list = element(tag.toLowerCase());
        body.append(list);
      }
      const item = element("li");
      appendInline(item, bullet[2], sources);
      list.append(item);
      continue;
    }
    list = null;
    const heading = /^#{1,6}\s+/.test(line);
    const row = element(heading ? "h4" : "p");
    appendInline(row, line.replace(/^#{1,6}\s+/, ""), sources);
    body.append(row);
  }
  return body;
}

function renderResult(data) {
  const sources = data.sources || [];
  const heading = element("div", null, "result-heading");
  const title = data.mode === "table" ? "查表结果" : data.mode === "answer" ? "回答" : "检索到的证据";
  heading.append(element("h3", title), element("span", data.route, "result-badge"));
  if (data.model) heading.append(element("span", data.model, "result-badge"));
  if (typeof data.matched === "number") heading.append(element("span", "适用 " + data.matched + " 条", "result-badge"));
  heading.append(element("span", sources.length + " 份来源", "result-badge"));
  result.replaceChildren(heading, renderAnswer(data.answer, sources));
  if (sources.length) {
    const sourceList = element("div", null, "source-list");
    for (const source of sources) {
      const disclosure = element("details");
      disclosure.append(element("summary", source.title + " · " + source.location));
      if (source.href) {
        const link = element("a", demo === "devices" ? "打开 PDF 原页 ↗" : "打开菜谱原文 ↗");
        link.href = source.href;
        link.target = "_blank";
        link.rel = "noopener";
        disclosure.append(link);
      }
      disclosure.append(element("pre", source.text));
      sourceList.append(disclosure);
    }
    result.append(sourceList);
  }
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  let question = input.value.trim();
  if (!question || busy) return;
  if (demo === "devices") {
    const mentioned = question.match(/DS18[A-Z0-9]+/i);
    if (!mentioned) {
      question = selectedModel + " " + question;
      input.value = question;
    } else if (["DS18B20", "DS18S20"].includes(mentioned[0].toUpperCase())) {
      selectModel(mentioned[0].toUpperCase());
    }
  }
  busy = true;
  submit.disabled = true;
  result.hidden = false;
  result.setAttribute("aria-busy", "true");
  result.replaceChildren(element("p", mode === "answer" ? "正在检索并生成回答…" : "正在检索，首次查询需要加载模型…", "muted loading"));
  try {
    const response = await fetch("/api/" + demo + "/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode }),
    });
    if (response.status === 401) {
      returnToLogin();
      return;
    }
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "查询失败，请稍后重试。");
    renderResult(data);
    showAvailability(data.remaining);
    if (data.remaining === 0) selectMode("preview");
  } catch (error) {
    result.replaceChildren(element("p", error.message || "连接失败，请检查演示是否仍在运行。", "error"));
  } finally {
    busy = false;
    submit.disabled = false;
    result.removeAttribute("aria-busy");
  }
});

input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault();
    form.requestSubmit();
  }
});

fetch("/api/catalog").then(response => {
  if (response.status === 401) {
    returnToLogin();
    return null;
  }
  if (!response.ok) throw new Error("连接失败");
  return response.json();
}).then(data => {
  if (!data) return;
  recipes = data.recipes;
  canGenerate = data.generation;
  viewer.textContent = data.viewer.display_name;
  viewer.title = data.viewer.display_name;
  viewer.hidden = false;
  signOut.hidden = false;
  showAvailability(data.remaining);
  renderRecipeCards();
}).catch(() => {
  availability.replaceChildren(element("span"), document.createTextNode("服务未连接"));
  const grid = document.getElementById("recipe-cards");
  if (grid) grid.replaceChildren(element("p", "暂时无法读取菜谱，请刷新页面重试。", "muted"));
});
