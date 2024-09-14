input = getDirectory("Choose input directory");
output = getDirectory("Choose where to save");

files = getFileList(input);
for (i=0; i<files.length; i++)
    oi(input, output, files[i]);

function oi (input, output, file) {
open(input + file);
run("Multiply...", "value=2.700");
selectWindow(file);
run("8-bit");
selectWindow(file);
setAutoThreshold("RenyiEntropy");
setOption("BlackBackground", false);
run("Convert to Mask");
selectWindow(file);
run("Analyze Particles...", "size=400-Infinity show=Masks in_situ");
selectWindow(file);
run("Dilate");
selectWindow(file);
run("Erode");
saveAs (".jpg", output + file);
run("Close");
}
waitForUser("TÁ PRONTO O SORVETINHO");