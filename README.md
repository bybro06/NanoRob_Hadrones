# Code for WRO 2024 Challenge - Team NANOROB HADRONES

This repository contains the Clever code developed by the **NANOROB HADRONES** team for the **WRO 2024** challenge. The project includes the solutions implemented for the **national finals**, the **world finals**, and the **Second Day Challenge**. It also features custom modules, integrated documentation, and additional tools to complement the robot's development.

---

## Project Structure

The repository is organized as follows:

- **`Help/`**: Custom help files that can replace those in Clever IDE to display detailed documentation for the modules we contributed.
- **`includes/`**: Auxiliary files containing specific movement functions.
- **`logos/`**: Images and graphic resources related to the team and project.
- **`modules/`**: Modules developed by the team for specific tasks, such as movements, trajectory calculations, and sensor management.
- **`Programas de reemplazo/`**: Pyhton and HTML programs used to change repeated words in the code without using other IDEs.
- **`robot/`**: Robot 3D model and instructions (only National robot yet).
- **`runWRO2024Nacional`**: Main program used in the **national finals**.
- **`runWRO2024Internacional`**: Main program used in the **world finals**.
- **`runWRO2024SecondDayChallenge`**: Main program designed for the **Second Day Challenge** of the **world finals**.

---

## Requirements

- **Clever IDE**: Use this development environment to program and upload the code to the robot.
- **EV3 Robot**: Configured according to the team's specifications to meet the challenges of WRO 2024.
- **Bluetooth or USB Connection**: To upload programs to the robot.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/NANOROB-HADRONES/WRO2024.git

2. Open Clever IDE and import the files from the folders as needed. An example of include inclusion is **`runWRO2024Nacional`**, and an example of module import is **`runWRO2024Internacional`**

3. Replace the Help folder in Clever IDE to enable integrated documentation for our modules. (See more details in the **Usage** section).

## Usage

1. **Selecting the Program File**  
   - **National Finals**: Use `runWRO2024Nacional`.  
   - **World Finals**: Use `runWRO2024Internacional`.  
   - **Second Day Challenge**: Use `runWRO2024SecondDayChallenge`.  

2. **Enabling Integrated Documentation**  
   The `Help` folder is designed to replace Clever IDE's default documentation and enhance it with details about our custom modules.  
   - Steps:
     1. Copy the contents of the `Help` folder from this repository.
     2. Replace the default `Help` folder in Clever IDE with this version.
     3. Restart Clever IDE to access the new documentation.

3. **Running the Programs**  
   - Connect your EV3 robot via **Bluetooth** or **USB**.
   - Load the selected program into Clever IDE.
   - Connect with your EV3 via Bluetooth or USB and compile the program selected.

4. **Reusing Modules**  
   The `modules` folder contains reusable components like movement algorithms, sensor handling, RGB algorithms and more.

---

## Contributions

This project is the work of the **NANOROB HADRONES** team, specifically for the WRO 2024 competition.  
We are not accepting contributions to ensure focus and integrity for the competition.

However, we welcome feedback, questions, or interest in our work! Feel free to contact us (details below) to engage with the robotics community.

---

## License

The contents of this repository are proprietary and intended solely for WRO 2024 use.  

- Redistribution, modification, or use outside the competition context is prohibited without prior authorization.  
- Unauthorized use will be considered a violation of intellectual property rights and competition regulations.

---

## Contact

For inquiries, collaborations, or questions about the **NANOROB HADRONES** team, reach out to us via:

- **Email**: [masquebots@gmail.com](mailto:masquebots@gmail.com)  
- **Instagram**: [@nanorobhadrones](https://instagram.com/nanorobhadrones)

We look forward to connecting with you and discussing robotics and the WRO competition!