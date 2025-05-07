// Mobile menu functionality
const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const navLinks = document.getElementById('navLinks');

if (mobileMenuBtn && navLinks) {
  mobileMenuBtn.addEventListener('click', () => {
    navLinks.classList.toggle('active');
    const icon = mobileMenuBtn.querySelector('i');
    if (navLinks.classList.contains('active')) {
      icon.classList.remove('fa-bars');
      icon.classList.add('fa-times');
    } else {
      icon.classList.remove('fa-times');
      icon.classList.add('fa-bars');
    }
  });
}

// Close mobile menu when clicking outside
document.addEventListener('click', (e) => {
  if (navLinks && mobileMenuBtn && 
      !navLinks.contains(e.target) && 
      !mobileMenuBtn.contains(e.target)) {
    navLinks.classList.remove('active');
    const icon = mobileMenuBtn.querySelector('i');
    icon.classList.remove('fa-times');
    icon.classList.add('fa-bars');
  }
});

// Authentication modals
document.addEventListener('DOMContentLoaded', function() {
  const loginBtn = document.getElementById('login-btn');
  const signupBtn = document.getElementById('signup-btn');
  const loginModal = document.getElementById('login-modal');
  const signupModal = document.getElementById('signup-modal');
  const showSignupLink = document.getElementById('show-signup');
  const showLoginLink = document.getElementById('show-login');
  
  if (loginBtn && loginModal) {
    loginBtn.addEventListener('click', () => {
      loginModal.style.display = 'block';
    });
  }
  
  if (signupBtn && signupModal) {
    signupBtn.addEventListener('click', () => {
      signupModal.style.display = 'block';
    });
  }
  
  if (showSignupLink && signupModal && loginModal) {
    showSignupLink.addEventListener('click', () => {
      loginModal.style.display = 'none';
      signupModal.style.display = 'block';
    });
  }
  
  if (showLoginLink && loginModal && signupModal) {
    showLoginLink.addEventListener('click', () => {
      signupModal.style.display = 'none';
      loginModal.style.display = 'block';
    });
  }
  
  // Close modals when clicking outside
  window.addEventListener('click', (e) => {
    if (e.target === loginModal) {
      loginModal.style.display = 'none';
    }
    if (e.target === signupModal) {
      signupModal.style.display = 'none';
    }
  });
  
  // Emergency contacts
  const contactForm = document.getElementById('contact-form');
  if (contactForm) {
    contactForm.addEventListener('submit', (e) => {
      e.preventDefault();
      
      const contactData = {
        name: document.getElementById('contact-name').value,
        type: document.getElementById('contact-type').value,
        phone: document.getElementById('contact-phone').value,
        email: document.getElementById('contact-email').value,
        notes: document.getElementById('contact-notes').value
      };
      
      fetch('/api/contacts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(contactData)
      })
      .then(response => response.json())
      .then(data => {
        // Add the new contact to the list
        addContactToList(data);
        // Reset the form
        contactForm.reset();
      })
      .catch(error => console.error('Error adding contact:', error));
    });
  }
  
  // Load emergency contacts
  function loadContacts() {
    fetch('/api/contacts')
      .then(response => response.json())
      .then(contacts => {
        const contactsList = document.getElementById('contacts-list');
        if (contactsList) {
          // Clear existing contacts
          contactsList.innerHTML = '';
          
          // Add each contact to the list
          contacts.forEach(contact => {
            addContactToList(contact);
          });
        }
      })
      .catch(error => console.error('Error loading contacts:', error));
  }
  
  // Add a contact to the list
  function addContactToList(contact) {
    const contactsList = document.getElementById('contacts-list');
    if (!contactsList) return;
    
    const contactCard = document.createElement('div');
    contactCard.className = 'contact-card';
    contactCard.dataset.id = contact.id;
    
    contactCard.innerHTML = `
      <div class="contact-header">
        <div class="contact-name">${contact.name}</div>
        <div class="contact-type">${contact.type}</div>
      </div>
      <div class="contact-details">
        <div><i class="fas fa-phone"></i> ${contact.phone}</div>
        ${contact.email ? `<div><i class="fas fa-envelope"></i> ${contact.email}</div>` : ''}
      </div>
      <div class="contact-actions">
        <button class="btn btn-outline btn-sm edit-contact" data-id="${contact.id}"><i class="fas fa-edit"></i> Edit</button>
        <button class="btn btn-outline btn-sm btn-danger delete-contact" data-id="${contact.id}"><i class="fas fa-trash"></i> Delete</button>
      </div>
    `;
    
    contactsList.appendChild(contactCard);
    
    // Add event listeners for edit and delete buttons
    const editBtn = contactCard.querySelector('.edit-contact');
    const deleteBtn = contactCard.querySelector('.delete-contact');
    
    if (editBtn) {
      editBtn.addEventListener('click', () => editContact(contact.id));
    }
    
    if (deleteBtn) {
      deleteBtn.addEventListener('click', () => deleteContact(contact.id));
    }
  }
  
  // Edit a contact
  function editContact(id) {
    // In a real app, this would open a modal with the contact's details
    // For now, we'll just log the ID
    console.log('Edit contact:', id);
  }
  
  // Delete a contact
  function deleteContact(id) {
    if (confirm('Are you sure you want to delete this contact?')) {
      fetch(`/api/contacts/${id}`, {
        method: 'DELETE'
      })
      .then(response => {
        if (response.ok) {
          // Remove the contact from the list
          const contactCard = document.querySelector(`.contact-card[data-id="${id}"]`);
          if (contactCard) {
            contactCard.remove();
          }
        }
      })
      .catch(error => console.error('Error deleting contact:', error));
    }
  }
  
  // Load contacts if on the emergency page
  if (document.querySelector('.emergency-page')) {
    loadContacts();
  }
  
  // Profile form submission
  const profileForm = document.getElementById('profile-form');
  if (profileForm) {
    profileForm.addEventListener('submit', (e) => {
      e.preventDefault();
      
      const profileData = {
        name: document.getElementById('profile-name').value,
        dob: document.getElementById('profile-dob').value,
        email: document.getElementById('profile-email').value,
        phone: document.getElementById('profile-phone').value,
        address: document.getElementById('profile-address').value
      };
      
      fetch('/api/profile', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(profileData)
      })
      .then(response => response.json())
      .then(data => {
        alert('Profile updated successfully');
      })
      .catch(error => console.error('Error updating profile:', error));
    });
  }
  
  // Health form submission
  const healthForm = document.getElementById('health-form');
  if (healthForm) {
    healthForm.addEventListener('submit', (e) => {
      e.preventDefault();
      
      const healthData = {
        bloodType: document.getElementById('blood-type').value,
        genotype: document.getElementById('genotype').value,
        allergies: document.getElementById('allergies').value,
        conditions: document.getElementById('conditions').value
      };
      
      fetch('/api/health', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(healthData)
      })
      .then(response => response.json())
      .then(data => {
        alert('Health information updated successfully');
      })
      .catch(error => console.error('Error updating health information:', error));
    });
  }
  
  // Settings form submissions
  const accountSettingsForm = document.getElementById('account-settings-form');
  if (accountSettingsForm) {
    accountSettingsForm.addEventListener('submit', (e) => {
      e.preventDefault();
      
      const accountData = {
        email: document.getElementById('email').value,
        password: document.getElementById('password').value,
        confirmPassword: document.getElementById('confirm-password').value
      };
      
      if (accountData.password && accountData.password !== accountData.confirmPassword) {
        alert('Passwords do not match');
        return;
      }
      
      fetch('/api/account', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(accountData)
      })
      .then(response => response.json())
      .then(data => {
        alert('Account settings updated successfully');
      })
      .catch(error => console.error('Error updating account settings:', error));
    });
  }
  
  const notificationSettingsForm = document.getElementById('notification-settings-form');
  if (notificationSettingsForm) {
    notificationSettingsForm.addEventListener('submit', (e) => {
      e.preventDefault();
      
      const notificationData = {
        crisisAlerts: document.querySelector('input[name="crisis-alerts"]').checked,
        medicationReminders: document.querySelector('input[name="medication-reminders"]').checked,
        appointmentReminders: document.querySelector('input[name="appointment-reminders"]').checked,
        weeklyReports: document.querySelector('input[name="weekly-reports"]').checked,
        frequency: document.getElementById('notification-frequency').value
      };
      
      fetch('/api/notifications', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(notificationData)
      })
      .then(response => response.json())
      .then(data => {
        alert('Notification settings updated successfully');
      })
      .catch(error => console.error('Error updating notification settings:', error));
    });
  }
  
  const privacySettingsForm = document.getElementById('privacy-settings-form');
  if (privacySettingsForm) {
    privacySettingsForm.addEventListener('submit', (e) => {
      e.preventDefault();
      
      const privacyData = {
        shareHealthData: document.querySelector('input[name="share-health-data"]').checked,
        shareCrisisAlerts: document.querySelector('input[name="share-crisis-alerts"]').checked,
        shareAnonymousData: document.querySelector('input[name="share-anonymous-data"]').checked,
        dataRetention: document.getElementById('data-retention').value
      };
      
      fetch('/api/privacy', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(privacyData)
      })
      .then(response => response.json())
      .then(data => {
        alert('Privacy settings updated successfully');
      })
      .catch(error => console.error('Error updating privacy settings:', error));
    });
  }
  
  const deviceSettingsForm = document.getElementById('device-settings-form');
  if (deviceSettingsForm) {
    deviceSettingsForm.addEventListener('submit', (e) => {
      e.preventDefault();
      
      const deviceData = {
        samplingRate: document.getElementById('sensor-sampling').value,
        batterySaver: document.getElementById('battery-saver').value,
        autoSync: document.querySelector('input[name="auto-sync"]').checked,
        backgroundMonitoring: document.querySelector('input[name="background-monitoring"]').checked
      };
      
      fetch('/api/device', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(deviceData)
      })
      .then(response => response.json())
      .then(data => {
        alert('Device settings updated successfully');
      })
      .catch(error => console.error('Error updating device settings:', error));
    });
  }
  
  // Data management actions
  const exportDataBtn = document.querySelector('.data-management-actions .btn-outline:nth-child(1)');
  if (exportDataBtn) {
    exportDataBtn.addEventListener('click', () => {
      fetch('/api/export-data')
        .then(response => response.blob())
        .then(blob => {
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = 'health-data.csv';
          document.body.appendChild(a);
          a.click();
          window.URL.revokeObjectURL(url);
          a.remove();
        })
        .catch(error => console.error('Error exporting data:', error));
    });
  }
  
  const clearDataBtn = document.querySelector('.data-management-actions .btn-outline:nth-child(2)');
  if (clearDataBtn) {
    clearDataBtn.addEventListener('click', () => {
      if (confirm('Are you sure you want to clear all your data? This action cannot be undone.')) {
        fetch('/api/clear-data', {
          method: 'POST'
        })
        .then(response => {
          if (response.ok) {
            alert('All data has been cleared');
          }
        })
        .catch(error => console.error('Error clearing data:', error));
      }
    });
  }
  
  const deleteAccountBtn = document.querySelector('.data-management-actions .btn-danger');
  if (deleteAccountBtn) {
    deleteAccountBtn.addEventListener('click', () => {
      if (confirm('Are you sure you want to delete your account? This action cannot be undone.')) {
        fetch('/api/delete-account', {
          method: 'POST'
        })
        .then(response => {
          if (response.ok) {
            window.location.href = '/';
          }
        })
        .catch(error => console.error('Error deleting account:', error));
      }
    });
  }
  
  // Sensor simulation
  const gsrInput = document.getElementById('gsrInput');
  const tempInput = document.getElementById('tempInput');
  const spo2Input = document.getElementById('spo2Input');
  const gsrInputValue = document.getElementById('gsrInputValue');
  const tempInputValue = document.getElementById('tempInputValue');
  const spo2InputValue = document.getElementById('spo2InputValue');
  
  if (gsrInput && tempInput && spo2Input) {
    gsrInput.addEventListener('input', () => {
      gsrInputValue.textContent = gsrInput.value;
    });
    
    tempInput.addEventListener('input', () => {
      tempInputValue.textContent = tempInput.value;
    });
    
    spo2Input.addEventListener('input', () => {
      spo2InputValue.textContent = spo2Input.value;
    });
    
    // Send data button
    const sendButton = document.getElementById('sendButton');
    if (sendButton) {
      sendButton.addEventListener('click', sendData);
    }
    
    // Auto send
    const autoButton = document.getElementById('autoButton');
    let autoInterval = null;
    
    if (autoButton) {
      autoButton.addEventListener('click', () => {
        if (autoInterval) {
          clearInterval(autoInterval);
          autoInterval = null;
          autoButton.textContent = 'Start Auto Send';
          autoButton.classList.remove('btn-primary');
          autoButton.classList.add('btn-outline');
        } else {
          autoInterval = setInterval(sendRandomData, 3000);
          autoButton.textContent = 'Stop Auto Send';
          autoButton.classList.remove('btn-outline');
          autoButton.classList.add('btn-primary');
        }
      });
    }
  }
  
  // Initialize Chart.js if on predictions page
  const chartCanvas = document.getElementById('timeSeriesChart');
  let chart;
  
  if (chartCanvas) {
    const ctx = chartCanvas.getContext('2d');
    chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'GSR',
            borderColor: 'rgb(255, 99, 132)',
            yAxisID: 'y',
            data: []
          },
          {
            label: 'Temperature',
            borderColor: 'rgb(255, 159, 64)',
            yAxisID: 'y1',
            data: []
          },
          {
            label: 'SpO2',
            borderColor: 'rgb(75, 192, 192)',
            yAxisID: 'y2',
            data: []
          },
          {
            label: 'Crisis Probability',
            borderColor: 'rgb(153, 102, 255)',
            yAxisID: 'y3',
            data: []
          }
        ]
      },
      options: {
        scales: {
          y: {
            type: 'linear',
            display: true,
            position: 'left',
            title: {
              display: true,
              text: 'GSR (µS)'
            },
            min: 0,
            max: 5
          },
          y1: {
            type: 'linear',
            display: true,
            position: 'right',
            title: {
              display: true,
              text: 'Temp (°C)'
            },
            min: 35,
            max: 39,
            grid: {
              drawOnChartArea: false
            }
          },
          y2: {
            type: 'linear',
            display: true,
            position: 'right',
            title: {
              display: true,
              text: 'SpO2 (%)'
            },
            min: 85,
            max: 100,
            grid: {
              drawOnChartArea: false
            }
          },
          y3: {
            type: 'linear',
            display: true,
            position: 'right',
            title: {
              display: true,
              text: 'Crisis Prob'
            },
            min: 0,
            max: 1,
            grid: {
              drawOnChartArea: false
            }
          }
        }
      }
    });
  }
  
  // Load database statistics
  loadStats();
  
  // Refresh stats button
  const refreshStatsButton = document.getElementById('refreshStats');
  if (refreshStatsButton) {
    refreshStatsButton.addEventListener('click', loadStats);
  }
  
  // Load recent data on startup
  fetch('/api/recent')
    .then(response => response.json())
    .then(data => {
      if (data.length > 0) {
        // Update with most recent reading
        const latest = data[data.length - 1];
        updateDashboard(
          {
            gsr: latest.gsr,
            temperature: latest.temperature,
            spo2: latest.spo2
          }, 
          {
            crisis_predicted: latest.crisis_predicted,
            crisis_probability: latest.crisis_probability
          }
        );
        
        // Update sliders
        if (gsrInput && tempInput && spo2Input) {
          gsrInput.value = latest.gsr;
          tempInput.value = latest.temperature;
          spo2Input.value = latest.spo2;
          gsrInputValue.textContent = latest.gsr.toFixed(1);
          tempInputValue.textContent = latest.temperature.toFixed(1);
          spo2InputValue.textContent = latest.spo2;
        }
      }
    })
    .catch(error => console.error('Error loading recent data:', error));
  
  // Functions
  function sendRandomData() {
    const gsr = Math.random() * 4.5 + 0.5;
    const temp = Math.random() * 4 + 35;
    const spo2 = Math.random() * 15 + 85;
    
    gsrInput.value = gsr.toFixed(1);
    tempInput.value = temp.toFixed(1);
    spo2Input.value = Math.round(spo2);
    
    gsrInputValue.textContent = gsr.toFixed(1);
    tempInputValue.textContent = temp.toFixed(1);
    spo2InputValue.textContent = Math.round(spo2);
    
    sendData();
  }
  
  function sendData() {
    const data = {
      gsr: parseFloat(gsrInput.value),
      temperature: parseFloat(tempInput.value),
      spo2: parseFloat(spo2Input.value)
    };
    
    fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
      updateDashboard(data, result);
      // Refresh stats after sending data
      loadStats();
    })
    .catch(error => console.error('Error:', error));
  }
  
  function updateDashboard(data, result) {
    // Update readings
    const gsrValue = document.getElementById('gsrValue');
    const tempValue = document.getElementById('tempValue');
    const spo2Value = document.getElementById('spo2Value');
    const crisisRisk = document.getElementById('crisisRisk');
    const riskDescription = document.getElementById('riskDescription');
    
    if (gsrValue && tempValue && spo2Value) {
      gsrValue.textContent = data.gsr.toFixed(2);
      tempValue.textContent = data.temperature.toFixed(1);
      spo2Value.textContent = data.spo2;
    }
    
    // Update risk display
    if (crisisRisk && riskDescription) {
      const riskPercentage = (result.crisis_probability * 100).toFixed(1) + '%';
      crisisRisk.textContent = riskPercentage;
      
      if (result.crisis_predicted === 1) {
        crisisRisk.style.color = 'var(--danger)';
        riskDescription.textContent = 'CRISIS ALERT! Immediate preventive action required.';
      } else if (result.crisis_probability > 0.75) {
        crisisRisk.style.color = 'var(--danger)';
        riskDescription.textContent = 'Very high risk - Immediate preventive action required';
      } else if (result.crisis_probability > 0.5) {
        crisisRisk.style.color = 'var(--warning)';
        riskDescription.textContent = 'High risk - Take preventive measures and contact your healthcare provider';
      } else if (result.crisis_probability > 0.25) {
        crisisRisk.style.color = 'var(--warning)';
        riskDescription.textContent = 'Moderate risk - Increase hydration and rest';
      } else {
        crisisRisk.style.color = 'var(--success)';
        riskDescription.textContent = 'Low risk - Continue normal activities with standard precautions';
      }
    }
    
    // Update chart if it exists
    if (chart) {
      const timestamp = new Date().toLocaleTimeString();
      chart.data.labels.push(timestamp);
      chart.data.datasets[0].data.push(data.gsr);
      chart.data.datasets[1].data.push(data.temperature);
      chart.data.datasets[2].data.push(data.spo2);
      chart.data.datasets[3].data.push(result.crisis_probability);
      
      // Keep only the most recent 20 points
      if (chart.data.labels.length > 20) {
        chart.data.labels.shift();
        chart.data.datasets.forEach(dataset => dataset.data.shift());
      }
      
      chart.update();
    }
  }
  
  function loadStats() {
    fetch('/api/stats')
      .then(response => response.json())
      .then(stats => {
        // Update statistics display
        const totalPredictions = document.getElementById('totalPredictions');
        const crisisPredictions = document.getElementById('crisisPredictions');
        const crisisPercentage = document.getElementById('crisisPercentage');
        const avgProbability = document.getElementById('avgProbability');
        const avgGsr = document.getElementById('avgGsr');
        const avgTemperature = document.getElementById('avgTemperature');
        const avgSpo2 = document.getElementById('avgSpo2');
        
        if (totalPredictions) totalPredictions.textContent = stats.total_predictions;
        if (crisisPredictions) crisisPredictions.textContent = stats.crisis_predictions;
        if (crisisPercentage) crisisPercentage.textContent = stats.crisis_percentage.toFixed(1) + '%';
        if (avgProbability) avgProbability.textContent = (stats.average_readings.crisis_probability * 100).toFixed(1) + '%';
        if (avgGsr) avgGsr.textContent = stats.average_readings.gsr.toFixed(2);
        if (avgTemperature) avgTemperature.textContent = stats.average_readings.temperature.toFixed(1) + '°C';
        if (avgSpo2) avgSpo2.textContent = stats.average_readings.spo2.toFixed(1) + '%';
      })
      .catch(error => console.error('Error loading stats:', error));
  }
  
  // Update crisis prediction
  function updatePrediction(probability) {
    const indicator = document.getElementById('crisis-indicator');
    const dot = indicator.querySelector('.indicator-dot');
    const text = indicator.querySelector('.indicator-text');
    const fill = document.getElementById('probability-fill');
    const probText = document.getElementById('probability-text');
    
    // Ensure probability is a valid number
    probability = parseFloat(probability) || 0;
    
    // Update probability meter
    const percentage = Math.min(Math.max(probability * 100, 0), 100);
    fill.style.width = `${percentage}%`;
    probText.textContent = `${percentage.toFixed(1)}%`;
    
    // Update indicator
    if (probability > 0.7) {
      dot.className = 'indicator-dot indicator-danger';
      text.textContent = `High Risk (${percentage.toFixed(1)}%)`;
      fill.style.backgroundColor = '#dc3545';
    } else if (probability > 0.4) {
      dot.className = 'indicator-dot indicator-warning';
      text.textContent = `Moderate Risk (${percentage.toFixed(1)}%)`;
      fill.style.backgroundColor = '#ffc107';
    } else {
      dot.className = 'indicator-dot indicator-normal';
      text.textContent = `Low Risk (${percentage.toFixed(1)}%)`;
      fill.style.backgroundColor = '#28a745';
    }
  }

  // Delete a medication
  async function handleDeleteMedication(medicationId, cardElement) {
    if (!confirm('Are you sure you want to delete this medication and all its history? This action cannot be undone.')) {
      return;
    }

    try {
      const response = await fetch(`/api/medications/${medicationId}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Medication deleted:', result.message);
        if (cardElement) {
          cardElement.remove();
        }
        // Refresh current medications list (optional, if not all cards are always visible)
        // await loadCurrentMedications(); 
        
        // Refresh Today's Schedule
        await loadTodaysSchedule();
        
        // Refresh the medication history chart
        await loadMedicationHistory();
        
        // Optionally, show a success message to the user
        // showToast('Medication deleted successfully.'); 
      } else {
        const errorData = await response.json();
        console.error('Error deleting medication:', errorData.error);
        alert(`Error deleting medication: ${errorData.error}`);
      }
    } catch (error) {
      console.error('Network error or other issue:', error);
      alert('An error occurred while deleting the medication. Please check your connection and try again.');
    }
  }
});
