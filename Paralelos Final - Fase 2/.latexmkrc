# Carpeta para auxiliares
$aux_dir = "build";
# Carpeta para PDF final
$out_dir = "build";
add_cus_dep('pdf', 'pdf', 0, 'copy_pdf');

sub copy_pdf {
    my ($base) = @_;
    my $pdf = "$out_dir/$base.pdf";
    my $dest = "$base.pdf";
    if (-e $pdf) {
        system("cp $pdf $dest");
    }
}
