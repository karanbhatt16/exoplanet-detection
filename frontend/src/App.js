import './App.css'
import React from 'react'
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import HomePage from './components/pages/HomePage';

export default function App() {
  return (
    <BrowserRouter>
      <div className='main-body'>
        <div className="orb orb-a"></div>
        <div className="orb orb-b"></div>
        <div className="orb orb-c"></div>

        <header className="hero panel">
            <div className="eyebrow">AI-assisted exoplanet transit analysis</div>
            <h1>Explore TESS light curves in a modern web dashboard.</h1>
            <p className="hero-copy">
                Search by TIC target ID or upload your own light-curve file, then inspect the strongest transit candidate,
                BLS period, estimated planet radius, and the full result dashboard.
            </p>
            <div className="hero-tags">
                <span>TIC target search</span>
                <span>Local file upload</span>
                <span>BLS periodogram</span>
                <span>Planet-radius estimate</span>
            </div>
        </header>

        <Routes>
          <Route path='/' element={<HomePage />} />
        </Routes>

      </div>
    </BrowserRouter>
  )
}
