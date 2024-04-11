import React, { useState } from 'react';
import Slider from '@mui/material/Slider';
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import axios from 'axios';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [n_keywords, setN] = useState(7); // Default value for n_keywords
  const [keywords, setKeywords] = useState('');
  const [udcs, setUdcs] = useState([]);

  const fetchKeywords = async () => {
    try {
      const keywordsRes = await axios.post('http://localhost:8000/keywords', { text, n_keywords });
      setKeywords(keywordsRes.data.keywords.join(', ')); // Join keywords into a string
    } catch (error) {
      console.error('Error fetching keywords:', error);
    }
  };

  const fetchUdcs = async () => {
    try {
      const udcsRes = await axios.post('http://localhost:8000/udcs', { text });
      setUdcs(udcsRes.data.udcs);
    } catch (error) {
      console.error('Error fetching UDCs:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setUdcs([]);
    fetchKeywords();
    fetchUdcs();
  };

  const handleSliderChange = (event, newValue) => {
    setN(newValue);
  };

  return (
    <div className="container">
      <div className="centered">
        <h1>Сервис для подбора УДК</h1>
        <form onSubmit={handleSubmit}>
          <div className="input-container">
            <p className="input-label">
              Введите текст аннотации вашей статьи:
            </p>
            <TextField
                id="outlined-multiline-static"
                value={text}
                multiline
                rows={8}
                sx={{ m: 1, width: '45%' }}
                variant="outlined"
                onChange={(e) => setText(e.target.value)} />
            <p className="input-label">
              Количество ключевых слов:
            </p>
            <div className="slider">
                <Slider
                  aria-label="Small steps"
                  value={n_keywords}
                  onChange={handleSliderChange} // Update n_keywords on slider change
                  step={1}
                  min={0}
                  max={15}
                  valueLabelDisplay="auto"
                />
            </div>
          </div>
          <Button variant="outlined" type='submit'>Отправить</Button>
        </form>
      </div>
      <div className="results">
        <div className="keywords">
          <h2>Ключевые слова</h2>

          <TextField
                id="outlined-multiline-static"
                value={keywords}
                multiline
                rows={8}
                sx={{ m: 1, width: '95%' }}
                variant="outlined"
                onChange={(e) => setKeywords(e.target.value)} />

        </div>
        <div className="udcs">
          <h2>Подходящие УДК</h2>
          <ul>
            {udcs.map(([item, link], index) => (
              <li key={index}>
                <a href={link}>{item}</a>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default App;
