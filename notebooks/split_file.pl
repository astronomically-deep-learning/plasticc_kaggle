#!/usr/bin/perl


# Written By: Jeff Andrews
# Last Edited: Dec 12, 2018

# This program is designed to split a text file
# into multiple files.
# 
# split_file.pl [in_file] [num_lines] 
#
# e.g. split_file.pl data.dat 1000


my $line,$filein,$fileout;
my $count,$count2;
my $temp;

$filein = $ARGV[0];
$num_lines = $ARGV[1]; 

open(IN,"./".$filein);

my $index = index($filein, "."); 
my $file_base = substr($filein, 0, $index); 
my $file_ext = substr($filein, $index); 

$total_counter = 0;


$objID = 0; 
$count = 0; 
$file_counter = 1; 
while($line=<IN>){
    $count ++;
    $total_counter ++; 


    @data = split(' ', $line); 

    $objID = $data[0]; 

    if($count > $num_lines && $objID != $objID_last){ 

        print "Written $total_counter lines\n"; 

        close OUT; 
        open(OUT,">".$file_base."_".$file_counter.$file_ext);  
        $file_counter ++; 
        $count = 0;

        print "./".$file_base."_".$file_counter.$file_ext; 
        print "\n"; 
    }  

    print OUT "$line"; 

    $objID_last = $objID; 

}

close OUT;
 






