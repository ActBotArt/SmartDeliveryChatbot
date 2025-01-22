const express = require('express');
const mongoose = require('mongoose');
const { WordTokenizer } = require('natural');
const tf = require('@tensorflow/tfjs-node');
const axios = require('axios');
const winston = require('winston');

// Настройка логгера
const logger = winston.createLogger({
  transports: [new winston.transports.File({ filename: 'chatbot.log' })],
});

// Подключение к MongoDB
mongoose.connect('mongodb://localhost:27017/chatbot', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

// Схема для истории диалогов
const dialogSchema = new mongoose.Schema({
  userId: String,
  message: String,
  intent: String,
  timestamp: { type: Date, default: Date.now },
});

const Dialog = mongoose.model('Dialog', dialogSchema);

// Инициализация Express
const app = express();
app.use(express.json());

// Загрузка модели машинного обучения
let model;
(async () => {
  model = await tf.loadLayersModel('file://./model/model.json');
  logger.info('ML model loaded');
})();

// Токенизация и предобработка текста
const tokenizer = new WordTokenizer();

function preprocessText(text) {
  const tokens = tokenizer.tokenize(text.toLowerCase());
  return tokens.filter(token => !/[0-9]/.test(token)); // Удаление чисел
}

// Классификация намерений
async function classifyIntent(text) {
  const tokens = preprocessText(text);
  const embeddings = await model.embed(tokens.join(' '));
  const prediction = model.predict(embeddings);
  const intents = ['delivery', 'payment', 'return'];
  return intents[prediction.argMax()];
}

// Интеграция с внешним API (пример: служба доставки)
async function getDeliveryStatus(orderId) {
  try {
    const response = await axios.get(`https://api.delivery.com/orders/${orderId}`);
    return response.data.status;
  } catch (error) {
    logger.error('Delivery API error:', error);
    return null;
  }
}

// Обработчик запросов
app.post('/api/message', async (req, res) => {
  const { userId, text } = req.body;

  try {
    // Классификация намерения
    const intent = await classifyIntent(text);

    // Генерация ответа
    let response;
    switch (intent) {
      case 'delivery':
        const orderId = text.match(/\d+/)?.[0];
        const status = await getDeliveryStatus(orderId);
        response = status ? `Статус заказа: ${status}` : 'Заказ не найден.';
        break;
      case 'payment':
        response = 'Оплата возможна картой или наличными.';
        break;
      default:
        response = 'Уточните ваш вопрос.';
    }

    // Сохранение в историю
    await Dialog.create({ userId, message: text, intent });

    res.json({ response });
  } catch (error) {
    logger.error('Processing error:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

// Запуск сервера
app.listen(3000, () => {
  logger.info('Server running on port 3000');
});