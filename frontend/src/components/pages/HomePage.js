import './homePage.css'
import React from 'react'

export default function HomePage() {
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

                    <form id="searchForm" className="search-form" encType="multipart/form-data">
                        <div className="source-switch">
                            <label className="source-option">
                                <input type="radio" name="source_mode" value="target" checked />
                                <span>TIC target ID</span>
                            </label>
                            <label className="source-option">
                                <input type="radio" name="source_mode" value="file" />
                                <span>Upload file</span>
                            </label>
                        </div>

                        <div className="field-grid">
                            <div className="field" data-mode-field="target">
                                <label for="targetId">TESS TIC ID</label>
                                <input id="targetId" name="target_id" type="text" placeholder="e.g. 251848941" />
                            </div>

                            <div className="field hidden" data-mode-field="file">
                                <label for="lightcurveFile">Light-curve file</label>
                                <input id="lightcurveFile" name="lightcurve_file" type="file" accept=".csv,.txt,.parquet,.fits,.fit,.lc" />
                            </div>

                            <div className="field">
                                <label for="sector">Sector</label>
                                <input id="sector" name="sector" type="number" placeholder="Optional" />
                            </div>

                            <div className="field check-field">
                                <label className="checkline">
                                    <input id="useAllSectors" name="use_all_sectors" type="checkbox" />
                                        <span>Use all sectors</span>
                                </label>
                            </div>

                            <div className="field">
                                <label for="minPeriod">Min period (days)</label>
                                <input id="minPeriod" name="min_period" type="number" step="0.0001" value="0.5" />
                            </div>

                            <div className="field">
                                <label for="maxPeriod">Max period (days)</label>
                                <input id="maxPeriod" name="max_period" type="number" step="0.0001" value="20.0" />
                            </div>

                            <div className="field">
                                <label for="minDuration">Min duration (hours)</label>
                                <input id="minDuration" name="min_duration" type="number" step="0.1" value="1.2" />
                            </div>

                            <div className="field">
                                <label for="maxDuration">Max duration (hours)</label>
                                <input id="maxDuration" name="max_duration" type="number" step="0.1" value="7.2" />
                            </div>

                            <div className="field">
                                <label for="durationSteps">Duration steps</label>
                                <input id="durationSteps" name="duration_steps" type="number" min="1" step="1" value="8" />
                            </div>

                            <div className="field">
                                <label for="stellarRadius">Stellar radius (Rsun)</label>
                                <input id="stellarRadius" name="stellar_radius" type="number" step="0.0001" placeholder="Optional" />
                            </div>

                            <div className="field">
                                <label for="stellarRadiusErr">Radius error (Rsun)</label>
                                <input id="stellarRadiusErr" name="stellar_radius_err" type="number" step="0.0001" placeholder="Optional" />
                            </div>
                        </div>

                        <div className="action-row">
                            <button id="runButton" type="submit" className="primary-button">Run Search</button>
                            <p className="action-help">The dashboard will appear below with a graph, score cards, and expandable details.</p>
                        </div>
                    </form>
                </section>

                <section id="loadingState" className="panel loading-panel hidden" aria-live="polite">
                    <div className="spinner"></div>
                    <div>
                        <h3>Analyzing the light curve</h3>
                        <p id="loadingText">Running BLS search and building the dashboard...</p>
                    </div>
                </section>

                <section id="errorBanner" className="panel error-panel hidden" aria-live="assertive"></section>

                <section id="emptyState" className="panel empty-panel">
                    <h2>Ready when you are</h2>
                    <p>
                        Enter a TIC target ID or upload a local light curve to generate the transit dashboard and candidate summary.
                    </p>
                </section>

                <section id="resultsSection" className="results-grid hidden">
                    <article className="panel plot-panel">
                        <div className="section-head">
                            <div>
                                <p className="section-kicker">Dashboard</p>
                                <h2>Transit analysis result</h2>
                            </div>
                            <div className="status-chip" id="resultStatusChip">No result yet</div>
                        </div>

                        <div className="plot-frame">
                            <img id="resultImage" alt="Transit analysis dashboard" />
                        </div>

                        <div className="details-toggle-row">
                            <button id="detailsToggle" className="secondary-button hidden" type="button">Show More Details</button>
                        </div>

                        <div id="detailsPanel" className="details-panel hidden">
                            <div className="detail-grid">
                                <section className="detail-card">
                                    <h3>Full Details</h3>
                                    <pre id="summaryText"></pre>
                                </section>
                                <section className="detail-card">
                                    <h3>Notes</h3>
                                    <ul id="notesList" className="list-block"></ul>
                                </section>
                                <section className="detail-card">
                                    <h3>Metrics</h3>
                                    <div id="metricsList" className="metrics-list"></div>
                                </section>
                            </div>
                        </div>
                    </article>

                    <aside className="panel side-panel">
                        <div className="section-head compact">
                            <div>
                                <p className="section-kicker">Summary</p>
                                <h2>Candidate at a glance</h2>
                            </div>
                        </div>

                        <div className="source-line" id="sourceLine">No search has been run yet.</div>

                        <div className="summary-grid">
                            <div className="summary-card">
                                <span>Confidence</span>
                                <strong id="confidenceValue">--</strong>
                            </div>
                            <div className="summary-card">
                                <span>Period</span>
                                <strong id="periodValue">--</strong>
                            </div>
                            <div className="summary-card">
                                <span>Duration</span>
                                <strong id="durationValue">--</strong>
                            </div>
                            <div className="summary-card">
                                <span>Radius</span>
                                <strong id="radiusValue">--</strong>
                            </div>
                        </div>

                        <div className="reason-block">
                            <h3>Status</h3>
                            <p id="resultReason">Run a search to see the closest transit candidate and the full explanation.</p>
                        </div>
                    </aside>
                </section>
            </div>
        </div>
    )
}