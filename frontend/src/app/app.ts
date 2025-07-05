import { HttpClient, HttpClientModule } from '@angular/common/http';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe, CommonModule  } from '@angular/common';
import { Footer } from './components/footer/footer';
import { Header } from './components/header/header';

@Component({
  selector: 'app-root',
  imports: [FormsModule, DecimalPipe, HttpClientModule, CommonModule, Footer, Header],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  constructor(private http: HttpClient) {

  }
  
  form = {
    lot_size: 0,
    year_built: 2000,
    beds: 3,
    baths: 2,
    property_type: 'Residential'
  };

  propertyTypes = [
    'Residential',
    'Residential Lease',
    'Residential Income',
    'Land',
    'Commercial Sale'
  ];

  result: {
    predicted_sale_price: number;
    confidence_value: number;
    actual_sale_price: number | null;
    actual_estimated_value: number | null;
    actual_confidence_value: number | null;
    error_vs_sale: number | null;
    error_vs_estimate: number | null;
  } | null = null;

  loading = false;
  error = '';

  predict() {
    this.loading = true;
    this.error = '';
    this.result = null;

    this.http.post('http://localhost:8000/predict', this.form)
      .subscribe({
        next: (res) => {
          this.result = res as any;
          this.loading = false;
        },
        error: (err) => {
          this.error = err.message || 'Prediction failed';
          this.loading = false;
        }
      });
  }
}
