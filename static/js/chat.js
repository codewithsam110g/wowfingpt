const chatInput = document.querySelector("#chat-input");
const sendButton = document.querySelector("#send-btn");
const chatContainer = document.querySelector(".chat-container");
const themeButton = document.querySelector("#theme-btn");
const deleteButton = document.querySelector("#delete-btn");

let userText = null;
const API_KEY = "PASTE-YOUR-API-KEY-HERE"; // Paste your API key here

const loadDataFromLocalstorage = () => {
  // Load saved chats and theme from local storage and apply/add on the page
  const themeColor = localStorage.getItem("themeColor");

  document.body.classList.toggle("light-mode", themeColor === "light_mode");
  themeButton.innerText = document.body.classList.contains("light-mode")
    ? "dark_mode"
    : "light_mode";

  const defaultText = `<div class="default-text">
                            <h1 style="font-family: 'Poppins', sans-serif;">Fin GPT</h1>
                            <p style="color:black;">Start a conversation and explore the power of AI.<br> Your chat history will be displayed here.</p>
                        </div>`;

  chatContainer.innerHTML = localStorage.getItem("all-chats") || defaultText;
  chatContainer.scrollTo(0, chatContainer.scrollHeight); // Scroll to bottom of the chat container
};

const createChatElement = (content, className) => {
  // Create new div and apply chat, specified class and set html content of div
  const chatDiv = document.createElement("div");
  chatDiv.classList.add("chat", className);
  chatDiv.innerHTML = content;
  return chatDiv; // Return the created chat div
};

const copyResponse = (copyBtn) => {
  // Copy the text content of the response to the clipboard
  const reponseTextElement = copyBtn.parentElement.querySelector("p");
  navigator.clipboard.writeText(reponseTextElement.textContent);
  copyBtn.textContent = "done";
  setTimeout(() => (copyBtn.textContent = "content_copy"), 1000);
};
let questionCounter = 0;
const MAX_QUESTIONS = 5;

// existing code...

const showPreparingAnimation = () => {
  const html = `<div class="chat-content">
                    <div class="chat-details">
                        <img src="static/img/logo1.png" alt="chatbot-img">
                        <div class="preparing-animation">
                            <span class="material-symbols-rounded">⏳</span>
                            <span class="animation-text">Analyzing...</span><br>
                            <span class="material-symbols-rounded">⏳</span>
                            <span class="animation-text">Reviving...</span><br>
                            <span class="material-symbols-rounded">⏳</span>
                            <span class="animation-text">Retriving...</span>
                        </div>
                    </div>
                    <span onclick="copyResponse(this)" class="material-symbols-rounded">content_copy</span>
                </div>`;
  const incomingChatDiv = createChatElement(html, "incoming");
  chatContainer.appendChild(incomingChatDiv);
  chatContainer.scrollTo(0, chatContainer.scrollHeight);
  return incomingChatDiv;
};


const getChatResponse = (incomingChatDiv) => {
  const pElement = document.createElement("p");

  $.ajax({
    url: "/chat/" + userText,
    type: "GET",
    success: function (response) {
      pElement.textContent = response;
      incomingChatDiv.querySelector(".preparing-animation").style.display =
        "none";
      incomingChatDiv.querySelector(".chat-details").appendChild(pElement);
      localStorage.setItem("all-chats", chatContainer.innerHTML);
      chatContainer.scrollTo(0, chatContainer.scrollHeight);
    },
    error: function (xhr, status, error) {
      pElement.classList.add("error");
      pElement.textContent =
        "Oops! Something went wrong while retrieving the response. Please try again.";
      incomingChatDiv.querySelector(".preparing-animation").style.display =
        "none";
      incomingChatDiv.querySelector(".chat-details").appendChild(pElement);
      localStorage.setItem("all-chats", chatContainer.innerHTML);
      chatContainer.scrollTo(0, chatContainer.scrollHeight);
    },
  });
};

const handleOutgoingChat = () => {
  userText = chatInput.value.trim();
  if (!userText) return;

  chatInput.value = "";
  chatInput.style.height = `${initialInputHeight}px`;

  questionCounter++;

  if (questionCounter > MAX_QUESTIONS) {
    const limitMessage = "You have completed your 5 questions for today.";
    const limitHtml = `<div class="chat-content">
                              <div class="chat-details">
                                  <img src="static/img/logo.jpg" alt="chatbot-img">
                                  <p class="bot">${limitMessage}</p>
                              </div>
                          </div>`;
    const limitDiv = createChatElement(limitHtml, "incoming");
    chatContainer.appendChild(limitDiv);
    chatContainer.scrollTo(0, chatContainer.scrollHeight);
    return;
  }
  const html = `<div class="chat-content">
                    <div class="chat-details">
                        <img src="static/img/user.jpg" alt="user-img">
                        <p>${userText}</p>
                    </div>
                </div>`;

  const outgoingChatDiv = createChatElement(html, "outgoing");
  chatContainer.querySelector(".default-text")?.remove();
  chatContainer.appendChild(outgoingChatDiv);
  chatContainer.scrollTo(0, chatContainer.scrollHeight);

  const incomingChatDiv = showPreparingAnimation();
  getChatResponse(incomingChatDiv);
};

// ... (rest of the code)

deleteButton.addEventListener("click", () => {
  // Remove the chats from local storage and call loadDataFromLocalstorage function
  if (confirm("Are you sure you want to delete all the chats?")) {
    localStorage.removeItem("all-chats");
    loadDataFromLocalstorage();
  }
});

themeButton.addEventListener("click", () => {
  // Toggle body's class for the theme mode and save the updated theme to the local storage
  document.body.classList.toggle("light-mode");
  localStorage.setItem("themeColor", themeButton.innerText);
  themeButton.innerText = document.body.classList.contains("light-mode")
    ? "dark_mode"
    : "light_mode";
});

const initialInputHeight = chatInput.scrollHeight;

chatInput.addEventListener("input", () => {
  // Adjust the height of the input field dynamically based on its content
  chatInput.style.height = `${initialInputHeight}px`;
  chatInput.style.height = `${chatInput.scrollHeight}px`;
});

chatInput.addEventListener("keydown", (e) => {
  // If the Enter key is pressed without Shift and the window width is larger
  // than 800 pixels, handle the outgoing chat
  if (e.key === "Enter" && !e.shiftKey && window.innerWidth > 800) {
    e.preventDefault();
    handleOutgoingChat();
  }
});

loadDataFromLocalstorage();
sendButton.addEventListener("click", handleOutgoingChat);


const endChatSession = () => {
  const endMessage = "Chat session has ended.";

  const endChatHtml = `
      <div class="chat-content">
          <div class="chat-details">
              <img src="static/img/logo.jpg" alt="chatbot-img">
              <div class="chat-text">
                  <p>${endMessage}</p>
              </div>
          </div>
      </div>`;

  const endChatDiv = createChatElement(endChatHtml, "incoming");
  chatContainer.appendChild(endChatDiv);
  chatContainer.scrollTo(0, chatContainer.scrollHeight);
};

// ... (rest of the code)

deleteButton.addEventListener("click", () => {
  if (confirm("Are you sure you want to delete all the chats?")) {
    localStorage.removeItem("all-chats");
    loadDataFromLocalstorage();
  }
});

// Event listener for End Chat button
const endChatButton = document.querySelector(".nav-menu #end-chat-btn");
endChatButton.addEventListener("click", endChatSession);

// ... (remaining code)
// spinner
// Spinner
var spinner = function () {
  setTimeout(function () {
      if ($('#spinner').length > 0) {
          $('#spinner').removeClass('show');
      }
  }, 1);
};
spinner();
$(window).on('load', function () {
    // Hide spinner when the page is fully loaded
    $('#spinner').removeClass('show');
});

  
  // Initiate the wowjs
  new WOW().init();


  // Sticky Navbar
  $(window).scroll(function () {
      if ($(this).scrollTop() > 45) {
          $('.navbar').addClass('sticky-top shadow-sm');
      } else {
          $('.navbar').removeClass('sticky-top shadow-sm');
      }
  });

  
  