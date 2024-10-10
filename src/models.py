class HybridModel(nn.Module):
    def __init__(self, transformer_model_name, num_labels, num_numeric_features):
        super(HybridModel, self).__init__()
        
        # Модель трансформера для текстов
        self.transformer = AutoModel.from_pretrained(transformer_model_name)
        
        # Полносвязный слой для обработки текстовых данных
        self.text_fc = nn.Linear(self.transformer.config.hidden_size, 128)
        
        # Полносвязный слой для числовых данных (и других нетекстовых признаков)
        self.numeric_fc = nn.Linear(num_numeric_features, 128)
        
        # Финальный классификационный слой
        self.classifier = nn.Linear(128 * 2, num_labels)
        
    def forward(self, text_input, numeric_input):
        # Обрабатываем текстовые данные через трансформер
        transformer_output = self.transformer(**text_input).pooler_output
        text_features = self.text_fc(transformer_output)
        
        # Обрабатываем числовые данные
        numeric_features = self.numeric_fc(numeric_input)
        
        # Объединяем текстовые и числовые признаки
        combined_features = torch.cat((text_features, numeric_features), dim=1)
        
        # Применяем финальный классификационный слой
        logits = self.classifier(combined_features)
        
        return logits

# Токенизируем текстовые данные с помощью токенизатора трансформера
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
text_encodings = tokenizer(list(dtr['text']), truncation=True, padding=True, return_tensors="pt")

# Преобразуем числовые/категориальные признаки (например, длину ролика, тип канала) в тензоры.
numeric_features = torch.tensor(dtr[['seconds', 'flag_closed', 'international']].values, dtype=torch.float32)

# Обучение модели: Создадим экземпляр гибридной модели и обучим её:
num_classes = 46
model = HybridModel('xlm-roberta-base', num_labels=num_classes, num_numeric_features=numeric_features.shape[1])

# Вычислим веса классов на основе их частоты в данных
le = LabelEncoder()
y_train = le.fit_transform(dtr['cartoon'])

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
# criterion = nn.CrossEntropyLoss()
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)


num_epochs = 5

# Пример цикла обучения
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    logits = model(text_encodings, numeric_features)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()