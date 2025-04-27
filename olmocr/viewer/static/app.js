// Interactive preview functionality
class PreviewManager {
  constructor() {
    this.currentPage = 1;
    this.totalPages = 0;
    this.initEventListeners();
    this.loadDocumentMeta();
  }

  async loadDocumentMeta() {
    try {
      const meta = await fetch('./document.json');
      const data = await meta.json();
      this.totalPages = data.page_count;
      this.updateNavigation();
      this.loadPageContent(this.currentPage);
    } catch (error) {
      console.error('Error loading document metadata:', error);
    }
  }

  updateNavigation() {
    document.querySelector('.document-info').textContent = 
      `Document • ${this.totalPages} pages`;
    
    const nav = document.querySelector('.page-nav');
    nav.innerHTML = `
      <button class="prev-page">← Previous</button>
      <span>Page ${this.currentPage} of ${this.totalPages}</span>
      <button class="next-page">→ Next</button>
    `;
  }

  async loadPageContent(pageNumber) {
    try {
      const response = await fetch(`./page_${pageNumber}.json`);
      const pageData = await response.json();
      this.renderPageContent(pageData);
    } catch (error) {
      console.error(`Error loading page ${pageNumber}:`, error);
    }
  }

  renderPageContent(data) {
    const contentContainer = document.querySelector('.text-content');
    contentContainer.innerHTML = `
      <div class="page-meta">
        <span>Page ${data.page_number}</span>
        <span>Confidence: ${(data.confidence * 100).toFixed(1)}%</span>
      </div>
      <div class="page-text">${data.text}</div>
    `;
  }

  initEventListeners() {
    document.addEventListener('click', (e) => {
      if (e.target.matches('.next-page')) {
        this.currentPage = Math.min(this.currentPage + 1, this.totalPages);
        this.loadPageContent(this.currentPage);
        this.updateNavigation();
      } else if (e.target.matches('.prev-page')) {
        this.currentPage = Math.max(this.currentPage - 1, 1);
        this.loadPageContent(this.currentPage);
        this.updateNavigation();
      }
    });
  }
}

// Initialize preview when DOM loads
document.addEventListener('DOMContentLoaded', () => {
  new PreviewManager();
});