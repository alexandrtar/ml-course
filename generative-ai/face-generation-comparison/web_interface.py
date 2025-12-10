import streamlit as st
import requests
import json
import base64
from PIL import Image
import io
import numpy as np
import traceback

# Настройки
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Face Generation Comparison",
    page_icon="🎭",
    layout="wide"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4F8BF9;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .error-box {
        background-color: #ffebee;
        border: 1px solid #ffcdd2;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=10)
def check_api_health():
    """Проверка доступности API с кэшированием"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

@st.cache_data(ttl=30)
def get_model_info():
    """Получение информации о моделях с кэшированием"""
    try:
        response = requests.get(f"{API_URL}/models", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def generate_images(model_type, num_images=1):
    """Генерация изображений"""
    try:
        data = {"model_type": model_type, "num_images": num_images}
        response = requests.post(f"{API_URL}/generate", json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            images = []
            
            for img_info in result.get("images", []):
                img_base64 = img_info.get("image", "")
                if not img_base64:
                    continue
                
                # Убираем префикс data URL
                if img_base64.startswith('data:image/png;base64,'):
                    img_base64 = img_base64.split(',')[1]
                
                try:
                    img_data = base64.b64decode(img_base64)
                    img = Image.open(io.BytesIO(img_data))
                    images.append(img)
                except Exception as img_e:
                    st.error(f"Ошибка декодирования: {img_e}")
                    continue
            
            return True, images, result
        else:
            return False, None, f"Ошибка {response.status_code}: {response.text}"
            
    except Exception as e:
        return False, None, f"Ошибка соединения: {str(e)}"

def interpolate_images(model_type, steps=10):
    """Интерполяция между изображениями"""
    try:
        data = {"model_type": model_type, "steps": steps}
        response = requests.post(f"{API_URL}/interpolate", json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            if "interpolation_grid" in result:
                img_base64 = result["interpolation_grid"]
                if img_base64.startswith('data:image/png;base64,'):
                    img_base64 = img_base64.split(',')[1]
                
                try:
                    img_data = base64.b64decode(img_base64)
                    grid_image = Image.open(io.BytesIO(img_data))
                    return True, grid_image, result
                except Exception as img_e:
                    return False, None, f"Ошибка декодирования: {img_e}"
            else:
                return False, None, "Нет данных изображения в ответе"
        else:
            return False, None, f"Ошибка {response.status_code}: {response.text}"
            
    except Exception as e:
        return False, None, f"Ошибка соединения: {str(e)}"

def main():
    """Основная функция веб-интерфейса"""
    st.markdown('<h1 class="main-header">🎭 Face Generation Comparison</h1>', unsafe_allow_html=True)
    
    # Проверка API
    with st.spinner("Проверка подключения к API..."):
        api_healthy, health_info = check_api_health()
    
    if not api_healthy:
        st.error("❌ API сервер недоступен")
        st.info("Запустите API сервер: `python start_api.py`")
        
        with st.expander("Подробная инструкция"):
            st.markdown("""
            1. Откройте терминал
            2. Активируйте окружение: `conda activate tf-gpu-env`
            3. Перейдите в папку проекта
            4. Запустите API: `python start_api.py`
            5. Обновите эту страницу (F5)
            """)
        
        if st.button("🔄 Обновить статус API"):
            st.rerun()
        return
    
    # Информация о моделях
    with st.spinner("Загрузка информации о моделях..."):
        models_loaded, model_info = get_model_info()
    
    if not models_loaded:
        st.error("❌ Не удалось загрузить информацию о моделях")
        st.button("🔄 Обновить", on_click=st.rerun)
        return
    
    # Отображение статуса
    st.success("✅ Система готова к работе")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("GAN Model")
        gan_status = model_info.get("gan", {}).get("status", "unknown")
        if gan_status == "loaded":
            st.success("✅ Загружена")
            st.info(f"Латентная размерность: {model_info['gan']['latent_dim']}")
            st.info(f"Размер изображения: {model_info['gan']['image_size'][0]}×{model_info['gan']['image_size'][1]}")
        else:
            st.warning("⚠️ Не загружена")
            st.info("Модель не обучена или повреждена")
    
    with col2:
        st.subheader("VAE Model")
        vae_status = model_info.get("vae", {}).get("status", "unknown")
        if vae_status == "loaded":
            st.success("✅ Загружена")
            st.info(f"Латентная размерность: {model_info['vae']['latent_dim']}")
            st.info(f"Размер изображения: {model_info['vae']['image_size'][0]}×{model_info['vae']['image_size'][1]}")
        else:
            st.warning("⚠️ Не загружена")
            st.info("Модель не обучена или повреждена")
    
    st.markdown("---")
    
    # Генерация изображений
    st.header("🖼️ Генерация изображений")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Модель:",
            ["gan", "vae"],
            format_func=lambda x: "GAN" if x == "gan" else "VAE"
        )
    
    with col2:
        num_images = st.slider(
            "Количество изображений:",
            1, 10, 1
        )
    
    if st.button("🎨 Сгенерировать", type="primary", use_container_width=True):
        with st.spinner(f"Генерация {num_images} изображений..."):
            success, images, result = generate_images(model_type, num_images)
        
        if success and images:
            st.success(f"✅ Сгенерировано {len(images)} изображений")
            
            # Отображение в сетке
            cols = st.columns(min(4, len(images)))
            for i, (col, img) in enumerate(zip(cols, images)):
                with col:
                    st.image(img, caption=f"Изображение {i+1}", use_container_width=True)
            
            # Сохранение
            if st.button("💾 Сохранить все"):
                import os
                os.makedirs("generated", exist_ok=True)
                for i, img in enumerate(images):
                    img.save(f"generated/{model_type}_{i+1}.png")
                st.success("✅ Изображения сохранены в папку 'generated/'")
        
        elif success and not images:
            st.warning("⚠️ API ответил успешно, но изображения не получены")
            with st.expander("Ответ API"):
                st.json(result)
        else:
            st.error(f"❌ Ошибка: {result}")
    
    st.markdown("---")
    
    # Интерполяция
    st.header("🔄 Интерполяция")
    
    col1, col2 = st.columns(2)
    
    with col1:
        interp_model = st.selectbox(
            "Модель для интерполяции:",
            ["gan", "vae"],
            key="interp_model",
            format_func=lambda x: "GAN" if x == "gan" else "VAE"
        )
    
    with col2:
        steps = st.slider(
            "Количество шагов:",
            3, 20, 10,
            key="interp_steps"
        )
    
    if st.button("🌀 Интерполировать", type="secondary", use_container_width=True):
        with st.spinner(f"Создание интерполяции..."):
            success, grid_image, result = interpolate_images(interp_model, steps)
        
        if success and grid_image:
            st.success("✅ Интерполяция создана")
            st.image(grid_image, use_container_width=True)
            
            if st.button("💾 Сохранить сетку"):
                import os
                os.makedirs("interpolation", exist_ok=True)
                grid_image.save(f"interpolation/{interp_model}_{steps}_steps.png")
                st.success(f"✅ Сохранено в 'interpolation/'")
        
        elif success and not grid_image:
            st.warning("⚠️ API ответил успешно, но сетка не получена")
            with st.expander("Ответ API"):
                st.json(result)
        else:
            st.error(f"❌ Ошибка: {result}")
    
    # Информация
    st.markdown("---")
    with st.expander("📊 Информация о системе"):
        st.json(health_info)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("❌ Критическая ошибка")
        with st.expander("Технические детали"):
            st.code(traceback.format_exc())
        
        st.info("""
        **Решение:**
        1. Проверьте, запущен ли API: `python start_api.py`
        2. Обновите страницу (F5)
        3. Проверьте наличие моделей в папке `trained_models/`
        """)