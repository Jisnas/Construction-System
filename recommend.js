document.addEventListener("DOMContentLoaded", function () {
    const suggestButton = document.getElementById("suggestButton");
    const budgetInput = document.getElementById("budgetInput");
    const resultContainer = document.getElementById("recommendation-results");

    const furniture_catalog = {
        "woodbed": 15000, 
        "doublebed": 20000, 
        "king_floorbed": 27000, 
        "upholsteredbed": 40000,
        "sofa": 10000, 
        "sofa1": 16000, 
        "bench1": 3000, 
        "dining1": 10000, 
        "dining2": 17000, 
        "dining3": 25000, 
        "table1": 7000, 
        "table2": 5000, 
        "wardrobe1": 15000, 
        "furniture1": 9000, 
        "furniture2": 8000, 
        "furniture3": 6000, 
        "furniture4": 8500, 
        "furniture5": 9000
    };

    const display_names = {
        "woodbed": "Wood Bed",
        "doublebed": "Double Bed",
        "king_floorbed": "King Floor Bed",
        "upholsteredbed": "Upholstered Bed",
        "sofa": "sofa",
        "sofa1": "corner sofa",
        "bench1": "bench",
        "dining1": "Rustic Wooden Dining Set",
        "dining2": "Vintage Cafe Dining Set",
        "dining3": "Modern Wooden Dining Set",
        "table1": "Wooden table",
        "table2": "EcoDine Set",
        "wardrobe1": "wardrobe",
        "furniture1": "dressing table",
        "furniture2": "recliner sofa",
        "furniture3": "couch",
        "furniture4": "armchair",
        "furniture5": "wooden desk"
    };

    suggestButton.addEventListener("click", async function () {
        const budget = parseFloat(budgetInput.value);
        
        if (!budget || budget <= 15000) {
            alert("Please enter a valid budget.");
            return;
        }

        try {
            const response = await fetch("http://127.0.0.1:5001/recommend", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ budget })
            });

            const data = await response.json();
            resultContainer.innerHTML = ""; 

            if (response.status !== 200) {
                resultContainer.innerText = "Error: " + data.error;
                return;
            }

            const list = document.createElement("ul");
            list.style.listStyleType = "none";
            list.style.padding = "0";

            data.recommendations.forEach(item => {
                const listItem = document.createElement("li");
                listItem.style.display = "flex";
                listItem.style.alignItems = "center";
                listItem.style.marginBottom = "10px";
                listItem.style.backgroundColor = "#f0f0f0";
                listItem.style.padding = "10px";
                listItem.style.borderRadius = "8px";
                listItem.style.cursor = "move";
                listItem.draggable = true;
                listItem.setAttribute('data-type', item);

                listItem.addEventListener('dragstart', (event) => {
                    event.dataTransfer.setData('text/plain', item);
                    console.log("Dragging:", item);
                });

                const icon = document.createElement("img");
                icon.src = `icons/${item}.png`;
                icon.alt = item;
                icon.style.width = "40px";
                icon.style.height = "40px";
                icon.style.marginRight = "10px";
                icon.draggable = false;

                const textContainer = document.createElement("div");
                textContainer.style.display = "flex";
                textContainer.style.flexDirection = "column";

                const itemName = document.createElement("span");
                itemName.innerText = display_names[item] || item;
                itemName.style.fontSize = "18px";
                itemName.style.marginBottom = "4px";
                itemName.style.userSelect = "none";

                const price = document.createElement("span");
                price.innerText = `₹${furniture_catalog[item].toLocaleString()}`;
                price.style.fontSize = "14px";
                price.style.color = "#666";
                price.style.userSelect = "none";

                textContainer.appendChild(itemName);
                textContainer.appendChild(price);

                listItem.appendChild(icon);
                listItem.appendChild(textContainer);
                list.appendChild(listItem);
            });

            resultContainer.appendChild(list);

            const helperMsg = document.createElement("p");
            helperMsg.textContent = "Drag recommended items to place them on your floor plan.";
            helperMsg.style.fontSize = "14px";
            helperMsg.style.fontStyle = "italic";
            helperMsg.style.marginTop = "10px";
            resultContainer.appendChild(helperMsg);

        } catch (error) {
            resultContainer.innerText = "Server error. Ensure the recommendation server is running.";
            console.error(error);
        }
    });
});
