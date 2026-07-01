import './homePage.css'
import React, { useState } from 'react'

function ResultSection({resultFetchStatus, analysisResult}) {
    const [showMoreDetails, setShowMoreDetails] = useState(false);

    const toggleMoreDetails = () => {
        setShowMoreDetails(!showMoreDetails);
    }

    return (
        <section id="resultsSection" className="results-grid">
            <article className="panel plot-panel">
                <div className="section-head">
                    <div>
                        <p className="section-kicker">Dashboard</p>
                        <h2>Transit analysis result</h2>
                    </div>
                    <div className="status-chip">{resultFetchStatus}</div>
                </div>

                <div className="plot-frame">
                    <img src={analysisResult["figure_data_uri"]} alt="Transit analysis dashboard" />
                </div>

                <div className="details-toggle-row">
                    <button className="secondary-button" type="button" onClick={toggleMoreDetails}>Show More Details</button>
                </div>

                {showMoreDetails ?
                    <div className="details-panel">
                        <div className="detail-grid">
                            <section className="detail-card">
                                <h3>Full Details</h3>
                                <pre id="summaryText">
                                    {analysisResult["summary_lines"].map((line, idx) => <p key={`smry-txt-ln-${idx}`}>{line}</p>)}
                                </pre>
                            </section>
                            <section className="detail-card">
                                <h3>Notes</h3>

                                <ul id="notesList" className="list-block">
                                    {analysisResult["notes"].map(line => <li>{line}</li>)}
                                </ul>
                            </section>
                            <section className="detail-card">
                                <h3>Metrics</h3>
                                <div id="metricsList" className="metrics-list">
                                    {Object.keys(analysisResult["metrics"]).map((matrix_key, idx) => {
                                        return (
                                            <span key={`note-ln-ukey-${idx}`}><b>{matrix_key}: </b>{analysisResult["metrics"][matrix_key]}</span>
                                        )
                                    })}
                                </div>
                            </section>
                        </div>
                    </div>
                    :
                    null
                }

            </article>

            <aside className="panel side-panel">
                <div className="section-head compact">
                    <div>
                        <p className="section-kicker">Summary</p>
                        <h2>Candidate at a glance</h2>
                    </div>
                </div>

                <div className="source-line">{analysisResult["source"] || 'No search has been run yet.'}</div>

                <div className="summary-grid">
                    <div className="summary-card">
                        <span>Confidence</span>
                        <strong>{`${analysisResult["confidence_percent"] || '--'}%`}</strong>
                    </div>
                    <div className="summary-card">
                        <span>Period</span>
                        <strong>{`${analysisResult.candidate?.period_days || '--'} days`}</strong>
                    </div>
                    <div className="summary-card">
                        <span>Duration</span>
                        <strong>{`${analysisResult.candidate?.duration_hours || '--'} hours`}</strong>
                    </div>
                    <div className="summary-card">
                        <span>Radius</span>
                        <strong>{analysisResult.candidate?.planet_radius_rearth || '--'}</strong>
                    </div>
                </div>

                <div className="reason-block">
                    <h3>Status</h3>
                    <p>{analysisResult["reason"] || 'Run a search to see the closest transit candidate and the full explanation.'}</p>
                </div>
            </aside>
        </section>
    )
}

export default function HomePage() {
    const [currentLayoutName, setCurrentLayoutName] = useState("target");
    const [analysisResult, setAnalysisResult] = useState({});
    const [resultFetchStatusText, setResultFetchStatusText] = useState("No result yet");
    const [isLoading, setIsLoading] = useState(false);
    
    const onSwitchLayout = (eve) => {
        setCurrentLayoutName(eve.target.value)
    }

    const onSubmitForm = async (eve) => {
        eve.preventDefault();

        const form_data = new FormData(eve.target);

        const data = Object.fromEntries(form_data.entries());
        console.log("Data being send to backend : ", data);

        setResultFetchStatusText("Loading...")
        setIsLoading(true);
        const response = await fetch("/analyze", { method: 'POST', body: form_data });
        if (response.status === 200) {
            setAnalysisResult(await response.json());
            setResultFetchStatusText("Result available");
        }
        setIsLoading(false);
    }

    return (
        <div>
            <div className="home-body">
                <section className="panel search-panel">
                    <div className="section-head">
                        <div>
                            <p className="section-kicker">Search</p>
                            <h2>Start a new analysis</h2>
                        </div>
                        <div className="status-chip" id="statusText">Ready</div>
                    </div>

                    <form className="search-form" onSubmit={onSubmitForm}>
                        <div className="source-switch">
                            <label className="source-option">
                                <input type="radio" name="source_mode" value="target" onChange={onSwitchLayout} checked={currentLayoutName === "target"} />
                                <span>TIC target ID</span>
                            </label>
                            <label className="source-option">
                                <input type="radio" name="source_mode" value="file" onChange={onSwitchLayout} checked={currentLayoutName === "file"} />
                                <span>Upload file</span>
                            </label>
                        </div>

                        <div className="field-grid">
                            {currentLayoutName === "file" ?
                                <div className="field" data-mode-field="file">
                                    <label htmlFor="lightcurveFile">Light-curve file</label>
                                    <input id="lightcurveFile" name="lightcurve_file" type="file" accept=".csv,.txt,.parquet,.fits,.fit,.lc" />
                                </div>
                                :
                                <div className="field" data-mode-field="target">
                                    <label htmlFor="targetId">TESS TIC ID</label>
                                    <input id="targetId" name="target_id" type="text" placeholder="e.g. 251848941" />
                                </div>
                            }

                            <div className="field">
                                <label htmlFor="sector">Sector</label>
                                <input id="sector" name="sector" type="number" placeholder="Optional" />
                            </div>

                            <div className="field check-field">
                                <label className="checkline">
                                    <input id="useAllSectors" name="use_all_sectors" type="checkbox" />
                                    <span>Use all sectors</span>
                                </label>
                            </div>

                            <div className="field">
                                <label htmlFor="minPeriod">Min period (days)</label>
                                <input id="minPeriod" name="min_period" type="number" step="0.0001" defaultValue="0.5" />
                            </div>

                            <div className="field">
                                <label htmlFor="maxPeriod">Max period (days)</label>
                                <input id="maxPeriod" name="max_period" type="number" step="0.0001" defaultValue="20.0" />
                            </div>

                            <div className="field">
                                <label htmlFor="minDuration">Min duration (hours)</label>
                                <input id="minDuration" name="min_duration" type="number" step="0.1" defaultValue="1.2" />
                            </div>

                            <div className="field">
                                <label htmlFor="maxDuration">Max duration (hours)</label>
                                <input id="maxDuration" name="max_duration" type="number" step="0.1" defaultValue="7.2" />
                            </div>

                            <div className="field">
                                <label htmlFor="durationSteps">Duration steps</label>
                                <input id="durationSteps" name="duration_steps" type="number" min="1" step="1" defaultValue="8" />
                            </div>

                            <div className="field">
                                <label htmlFor="stellarRadius">Stellar radius (Rsun)</label>
                                <input id="stellarRadius" name="stellar_radius" type="number" step="0.0001" placeholder="Optional" />
                            </div>

                            <div className="field">
                                <label htmlFor="stellarRadiusErr">Radius error (Rsun)</label>
                                <input id="stellarRadiusErr" name="stellar_radius_err" type="number" step="0.0001" placeholder="Optional" />
                            </div>
                        </div>

                        <div className="action-row">
                            <button type="submit" className="primary-button">Run Search</button>
                            <p className="action-help">The dashboard will appear below with a graph, score cards, and expandable details.</p>
                        </div>
                    </form>
                </section>
                
                {isLoading ? 
                    <section className="panel loading-panel" aria-live="polite">
                        <div className="spinner"></div>
                        <div>
                            <h3>Analyzing the light curve</h3>
                            <p id="loadingText">Running BLS search and building the dashboard...</p>
                        </div>
                    </section>
                : null}
                
                {/* <section id="errorBanner" className="panel error-panel" aria-live="assertive"></section> */}

                {/* <section id="emptyState" className="panel empty-panel">
                    <h2>Ready when you are</h2>
                    <p>
                        Enter a TIC target ID or upload a local light curve to generate the transit dashboard and candidate summary.
                    </p>
                </section> */}

                {
                Object.keys(analysisResult).length ? 
                    <ResultSection analysisResult={analysisResult} resultFetchStatus={resultFetchStatusText} />
                : null }
            </div>
        </div>
    )
}