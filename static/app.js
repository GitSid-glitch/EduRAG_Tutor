const state = {
  topic: null,
  history: [],
};

const pdfInput = document.getElementById("pdfInput");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");
const topicCard = document.getElementById("topicCard");
const emptyState = document.getElementById("emptyState");
const chatSection = document.getElementById("chatSection");
const messages = document.getElementById("messages");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");

function setStatus(text) {
  uploadStatus.textContent = text;
}

function renderTopic(topic) {
  topicCard.classList.remove("hidden");
  topicCard.innerHTML = `
    <p class="eyebrow">Indexed Topic</p>
    <h3>${topic.title}</h3>
    <p>${topic.summary || "Document indexed and ready for chat."}</p>
    <div class="stats">
      <span>${topic.pages} pages</span>
      <span>${topic.chunks} chunks</span>
      <span>${topic.embeddingBackend}</span>
    </div>
  `;
}

function renderMessage(role, html) {
  const node = document.createElement("article");
  node.className = `message ${role}`;
  node.innerHTML = html;
  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
}

function renderSources(sources) {
  if (!sources?.length) return "";
  return `
    <details class="sources">
      <summary>Supporting passages</summary>
      <div class="sources-body">
        ${sources
          .map(
            (source) => `
            <div class="source-card">
              <div class="source-meta">From page ${source.page}</div>
              <p>${source.excerpt}</p>
            </div>
          `
          )
          .join("")}
      </div>
    </details>
  `;
}

function renderImage(image) {
  if (!image) return "";
  return `
    <div class="image-card">
      <img src="${image.filename}" alt="${image.title}" />
      <div>
        <p class="eyebrow">Relevant Visual</p>
        <h4>${image.title}</h4>
        <p>${image.description}</p>
      </div>
    </div>
  `;
}

uploadBtn.addEventListener("click", async () => {
  const file = pdfInput.files[0];
  if (!file) {
    setStatus("Select a PDF first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  setStatus("Extracting text, chunking, and building embeddings...");

  const response = await fetch("/upload", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Upload failed." }));
    setStatus(error.detail || "Upload failed.");
    return;
  }

  state.topic = await response.json();
  state.history = [];
  renderTopic(state.topic);
  emptyState.classList.add("hidden");
  chatSection.classList.remove("hidden");
  messages.innerHTML = "";
  renderMessage(
    "assistant",
    `<div><p class="eyebrow">Tutor Ready</p><p>Your document is indexed. Ask a question and I’ll answer using retrieved evidence only.</p></div>`
  );
  setStatus(`Ready. Indexed ${state.topic.chunks} chunks from ${state.topic.title}.`);
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = questionInput.value.trim();
  if (!message || !state.topic) return;

  renderMessage("user", `<p>${message}</p>`);
  state.history.push({ role: "user", content: message });
  questionInput.value = "";

  const formData = new FormData();
  formData.append(
    "payload",
    JSON.stringify({
      topicId: state.topic.topicId,
      message,
      history: state.history,
    })
  );

  renderMessage("assistant", `<p class="typing">Thinking with retrieved chunks...</p>`);
  const pending = messages.lastElementChild;

  const response = await fetch("/chat", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    pending.innerHTML = "<p>Something went wrong while generating the answer.</p>";
    return;
  }

  const data = await response.json();
  state.history.push({ role: "assistant", content: data.answer });
  pending.innerHTML = `
    <div class="answer-block">
      <div class="answer-panel">
        <p class="eyebrow">Answer</p>
        <div class="answer-text">${data.answer}</div>
      </div>
      ${renderImage(data.image)}
      ${renderSources(data.sources)}
    </div>
  `;
  messages.scrollTop = messages.scrollHeight;
});
