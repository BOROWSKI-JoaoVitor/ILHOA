// Get input and output directories
input = getDirectory("Choose input directory");
output = getDirectory("Select where to save");

// Main function
function main(directory) {
    // Validate directory and list files
    if (directory == "") {
        print("Error: No directory selected.");
        return;
    }
    
    files = getFileList(directory);
    if (files.length == 0) {
        print("Error: No files in the directory.");
        return;
    }
    
    sortedFiles = Array.sort(files);
    maxIndex = sortedFiles.length;
    
    index = 0;
    while (index < maxIndex) {
        // Split terms for the first file in the sorted list
        terms = split(sortedFiles[index], "_");
        if (terms.length < 2) {
            print("Error: Unexpected file naming in " + sortedFiles[index]);
            index++;
            continue;
        }
        
        // Define criteria terms from the first file
        CTerm1 = terms[0];
        CTerm2 = terms[1];
        
        // Process batch based on criteria terms
        result = auxiliar(index, sortedFiles, CTerm1, CTerm2);
        rstLgt = result.length -1;
        batch = Array.deleteIndex(result, rstLgt);
        index = result[rstLgt];
        
        // Merge if there are files in batch
        if (batch.length > 0) {
            merge(batch);
        }
    }
}

// Auxiliary function: Collects files into a batch based on matching terms
function auxiliar(startIndex, files, CTerm1, CTerm2) {
    batch = newArray();
    i = startIndex;
    
    // Loop to collect files matching CTerm1 and CTerm2
    while (i < files.length) {
        terms = split(files[i], "_");
        if (terms.length < 2) break;
        
        if (terms[0] == CTerm1 && terms[1] == CTerm2) {
            batch = Array.concat(batch, files[i]);
        } else {
            break; // Stop if criteria terms do not match
        }
        
        i++;
    }
    
    return Array.concat(batch, i); // Return batch and updated index
}

// Mock merge function
function merge(batch) {
	images = toLowerCase(String.join(batch, " "));
	terms = split(batch[0], "_");
    Term1 = terms[0];
    Term2 = terms[1];
    mergedImageName = Term1 + "_" + Term2 + ".jpg";

    run("Grid/Collection stitching", 
    "type=[Sequential Images] " + 
    "order=[All files in directory] " + 
    "directory=[" + input + "] " + 
    "confirm_files " + 
    "output_textfile_name=TileConfiguration.txt " + 
    "fusion_method=[Linear Blending] " + 
    "regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 frame=1 " + 
    "computation_parameters=[Save computation time (but use more RAM)] " + 
    "image_output=[Fuse and display] " + 
    "output_directory=[" + output + "] " + 
    images);
    saveAs(".jpg", output + mergedImageName);
    run("Close");
}

// Run the main function
main(input);